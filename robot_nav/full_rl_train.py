import numpy as np
from dev_sim import DEV_SIM
from ignite.engine import Engine, Events
import hydra
from omegaconf import DictConfig
from utils.utils import setup_clearm
from utils.full_utils import evaluate, attach_logging, init_checkpoint


@hydra.main(version_base=None, config_path="configs", config_name="full_rl_config")
def main(cfg: DictConfig):
    """Main training function"""

    clearml_logger = setup_clearm(cfg.clearml)

    base_model = hydra.utils.instantiate(cfg.base_model)
    deviation_model = hydra.utils.instantiate(cfg.dev_model)

    sim = DEV_SIM(
        world_file="robot_world.yaml", disable_plotting=False
    )  # instantiate environment

    dev_replay_buffer = hydra.utils.instantiate(cfg.replay_buffer)
    base_replay_buffer = hydra.utils.instantiate(cfg.replay_buffer)

    def run_step(engine, timestep):
        observation = engine.state.observation
        switch: bool = bool(engine.state.epoch % 2)
        state, terminal = base_model.prepare_state(observation)
        action = base_model.get_action(
            np.array(state), True
        )  # get an action from the model
        dev_state, terminal = deviation_model.prepare_state(observation, action)
        override_action = deviation_model.get_action(np.array(dev_state), ~switch)
        next_observation = sim.step(
            lin_velocity=action[0],
            ang_velocity=action[1],
            override_lin=override_action[0],
            override_ang=override_action[1],
            switch=switch,
        )  # get data from the environment
        if switch:
            next_state, terminal = base_model.prepare_state(next_observation)
            base_replay_buffer.add(
                state, action, next_observation["reward"], terminal, next_state
            )
        else:
            next_state, _ = base_model.prepare_state(next_observation)
            action = base_model.get_action(np.array(next_state), True)
            next_dev_state, terminal = deviation_model.prepare_state(
                next_observation, action
            )
            dev_replay_buffer.add(
                dev_state,
                override_action,
                next_observation["reward"],
                terminal,
                next_dev_state,
            )
        if terminal:
            engine.terminate_epoch()
            engine.state.timestep = timestep

        engine.state.observation = next_observation

    trainer = Engine(run_step)

    @trainer.on(Events.EPOCH_STARTED)
    def reset_environment():
        trainer.state.observation = sim.reset()
        trainer.state.timestep = 0

    @trainer.on(Events.EPOCH_COMPLETED(every=cfg.train_every_n))
    def update_model(engine):
        base_log = base_model.train(
            replay_buffer=base_replay_buffer,
            iterations=cfg.training_iterations,
            batch_size=cfg.batch_size,
        )
        dev_log = deviation_model.train(
            replay_buffer=dev_replay_buffer,
            iterations=cfg.training_iterations,
            batch_size=cfg.batch_size,
        )
        engine.state.output = base_log
        engine.state.dev_output = dev_log

    @trainer.on(Events.EPOCH_COMPLETED(every=cfg.episodes_per_epoch))
    def evaluate_model():
        trainer.state.base_eval = evaluate(
            base_model, trainer.state.epoch, sim, eval_episodes=cfg.nr_eval_episodes
        )
        trainer.state.dev_eval = evaluate(
            base_model,
            trainer.state.epoch,
            sim,
            eval_episodes=cfg.nr_eval_episodes,
            eval_dev=True,
            dev_model=deviation_model,
        )

    attach_logging(clearml_logger, trainer, cfg)
    init_checkpoint(trainer, base_model, deviation_model, cfg)
    trainer.run(epoch_length=cfg.max_steps, max_epochs=cfg.max_epochs)
    clearml_logger.close()


if __name__ == "__main__":
    main()
