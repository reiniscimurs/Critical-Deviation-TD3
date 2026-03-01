from utils.dev_utils import attach_logging, evaluate, init_checkpoint

import numpy as np
from dev_sim import DEV_SIM
from ignite.engine import Engine, Events
import hydra
from omegaconf import DictConfig
from utils.utils import setup_clearm, compute_action


@hydra.main(version_base=None, config_path="configs", config_name="dev_rl_config")
def main(cfg: DictConfig):
    """Main training function"""
    clearml_logger = setup_clearm(cfg.clearml)

    model = hydra.utils.instantiate(cfg.dev_model)

    sim = DEV_SIM(world_file="robot_world.yaml", disable_plotting=False)
    replay_buffer = hydra.utils.instantiate(cfg.replay_buffer)

    def run_step(engine, timestep):
        observation = engine.state.observation
        a = compute_action(
            observation["distance"], observation["sin"], observation["cos"]
        )
        state, terminal = model.prepare_state(observation, a)

        action = model.get_action(np.array(state), True)  # get an action from the model

        next_observation = sim.step(
            lin_velocity=a[0],
            ang_velocity=a[1],
            override_lin=action[0],
            override_ang=action[1],
            switch=False,
        )  # get data from the environment
        a = compute_action(
            next_observation["distance"],
            next_observation["sin"],
            next_observation["cos"],
        )
        next_state, terminal = model.prepare_state(
            next_observation, a
        )  # get a next state representation
        replay_buffer.add(
            state, action, next_observation["reward"], terminal, next_state
        )  # add experience to the replay buffer
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
        log = model.train(
            replay_buffer=replay_buffer,
            iterations=cfg.training_iterations,
            batch_size=cfg.batch_size,
        )
        engine.state.output = log

    @trainer.on(Events.EPOCH_COMPLETED(every=cfg.episodes_per_epoch))
    def evaluate_model():
        trainer.state.eval = evaluate(
            model, trainer.state.epoch, sim, eval_episodes=cfg.nr_eval_episodes
        )

    attach_logging(clearml_logger, trainer, cfg)
    init_checkpoint(trainer, model, cfg)
    trainer.run(epoch_length=cfg.max_steps, max_epochs=cfg.max_epochs)
    clearml_logger.close()


if __name__ == "__main__":
    main()
