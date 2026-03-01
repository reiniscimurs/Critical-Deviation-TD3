from ignite.engine import Engine, Events
import numpy as np
from utils.utils import setup_clearm
import hydra
from omegaconf import DictConfig
from utils.dual_utils import attach_logging, evaluate, init_checkpoint


@hydra.main(version_base=None, config_path="configs", config_name="dual_rl_config")
def main(cfg: DictConfig):
    """Main training function"""

    clearml_logger = setup_clearm(cfg.clearml)
    train_model = hydra.utils.instantiate(cfg.base_model)

    sim = hydra.utils.instantiate(cfg.sim)

    train_replay_buffer = hydra.utils.instantiate(cfg.replay_buffer)

    def run_step(engine, timestep):
        observation = engine.state.observation
        state, terminal = train_model.prepare_state(observation)
        action, dev_action = train_model.get_action(state, True)
        next_observation = sim.step(
            lin_velocity=action[0],
            ang_velocity=action[1],
            override_lin=dev_action[0],
            override_ang=dev_action[1],
        )  # get data from the environment
        next_state, terminal = train_model.prepare_state(next_observation)
        train_replay_buffer.add(
            state,
            np.concatenate((action, dev_action)),
            next_observation["reward"],
            terminal,
            next_state,
        )
        engine.state.observation = next_observation

        if terminal:
            engine.terminate_epoch()
            engine.state.timestep = timestep

    trainer = Engine(run_step)

    @trainer.on(Events.EPOCH_STARTED)
    def reset_environment():
        trainer.state.observation = sim.reset()
        trainer.state.timestep = 0

    @trainer.on(Events.EPOCH_COMPLETED(every=cfg.train_every_n))
    def update_model(engine):
        log = train_model.train(
            replay_buffer=train_replay_buffer,
            iterations=cfg.training_iterations,
            batch_size=cfg.batch_size,
        )
        engine.state.output = log

    @trainer.on(Events.EPOCH_COMPLETED(every=cfg.episodes_per_epoch))
    def evaluate_model():
        trainer.state.eval = evaluate(
            train_model, trainer.state.epoch, sim, eval_episodes=cfg.nr_eval_episodes
        )

    attach_logging(clearml_logger, trainer, cfg)
    init_checkpoint(trainer, train_model, cfg)
    trainer.run(epoch_length=cfg.max_steps, max_epochs=cfg.max_epochs)
    clearml_logger.close()


if __name__ == "__main__":
    main()
