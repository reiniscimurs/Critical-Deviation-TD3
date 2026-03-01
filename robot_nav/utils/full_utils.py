from pathlib import Path

import numpy as np
import torch
from ignite.engine import Events
from ignite.handlers import Checkpoint, DiskSaver
from ignite.handlers.clearml_logger import OutputHandler


def evaluate(model, epoch, sim, eval_episodes=10, eval_dev=False, dev_model=None):
    print("..............................................")
    print(f"Epoch {epoch}. Evaluating scenarios")
    avg_reward = 0.0
    col = 0
    goals = 0
    for _ in range(eval_episodes):
        count = 0

        observation = sim.reset()
        done = False
        while not done and count < 301:
            state, terminal = model.prepare_state(observation)
            action = model.get_action(np.array(state), False)
            if eval_dev:
                dev_state, terminal = dev_model.prepare_state(observation, action)
                override_action = dev_model.get_action(np.array(dev_state), False)
            else:
                override_action = [0, 0]
            observation = sim.step(
                lin_velocity=action[0],
                ang_velocity=action[1],
                override_lin=override_action[0],
                override_ang=override_action[1],
                switch=~eval_dev,
            )
            avg_reward += observation["reward"]
            count += 1
            if observation["collision"]:
                col += 1
            if observation["goal"]:
                goals += 1
            done = observation["collision"] or observation["goal"]
    avg_reward /= eval_episodes
    avg_col = col / eval_episodes
    avg_goal = goals / eval_episodes
    print(f"Average Reward: {avg_reward}")
    print(f"Average Collision rate: {avg_col}")
    print(f"Average Goal rate: {avg_goal}")
    print("..............................................")
    eval_result = {"avg_reward": avg_reward, "avg_col": avg_col, "avg_goal": avg_goal}
    return eval_result


def attach_logging(clearml_logger, trainer, cfg):
    clearml_logger.attach_output_handler(
        trainer,
        tag="base_training",
        event_name=Events.EPOCH_COMPLETED(every=cfg.train_every_n),
        output_transform=lambda out: out,
    )
    clearml_logger.attach(
        trainer,
        log_handler=OutputHandler(
            tag="base_eval",
            metric_names=["avg_reward", "avg_col", "avg_goal"],
            output_transform=lambda _: {
                "avg_reward": trainer.state.base_eval["avg_reward"],
                "avg_col": trainer.state.base_eval["avg_col"],
                "avg_goal": trainer.state.base_eval["avg_goal"],
            },
        ),
        event_name=Events.EPOCH_COMPLETED(every=cfg.episodes_per_epoch),
    )

    clearml_logger.attach(
        trainer,
        log_handler=OutputHandler(
            tag="dev_eval",
            metric_names=["avg_reward", "avg_col", "avg_goal"],
            output_transform=lambda _: {
                "avg_reward": trainer.state.dev_eval["avg_reward"],
                "avg_col": trainer.state.dev_eval["avg_col"],
                "avg_goal": trainer.state.dev_eval["avg_goal"],
            },
        ),
        event_name=Events.EPOCH_COMPLETED(every=cfg.episodes_per_epoch),
    )

    clearml_logger.attach(
        trainer,
        log_handler=OutputHandler(
            tag="dev_training",
            output_transform=lambda _: {
                "loss": trainer.state.dev_output["loss"],
                "avg_Q": trainer.state.dev_output["avg_Q"],
                "max_Q": trainer.state.dev_output["max_Q"],
            },
        ),
        event_name=Events.EPOCH_COMPLETED(every=cfg.train_every_n),
    )


def init_checkpoint(trainer, base_model, dev_model, cfg):
    to_save = {
        "base_actor": base_model.actor,
        "base_actor_target": base_model.actor_target,
        "base_critic": base_model.critic,
        "base_critic_target": base_model.critic_target,
        "base_actor_optimizer": base_model.actor_optimizer,
        "base_critic_optimizer": base_model.critic_optimizer,
        "dev_actor": dev_model.actor,
        "dev_actor_target": dev_model.actor_target,
        "dev_critic": dev_model.critic,
        "dev_critic_target": dev_model.critic_target,
        "dev_actor_optimizer": dev_model.actor_optimizer,
        "dev_critic_optimizer": dev_model.critic_optimizer,
        "trainer": trainer,
    }

    save_dir = Path(cfg.save_directory)
    save_dir.mkdir(parents=True, exist_ok=True)

    if cfg.load_checkpoint:
        checkpoint_fp = save_dir / cfg.checkpoint
        checkpoint = torch.load(checkpoint_fp, map_location=cfg.device)

        Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

        print(f"Loaded checkpoint from {checkpoint_fp}")

    handler = Checkpoint(
        to_save,
        DiskSaver(save_dir, require_empty=False),
        n_saved=1,
        filename_prefix="rl",
    )

    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=cfg.save_every), handler)
