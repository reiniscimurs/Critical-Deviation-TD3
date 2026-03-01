from pathlib import Path

import statistics
import torch
from ignite.engine import Events
from ignite.handlers import Checkpoint, DiskSaver
from ignite.handlers.clearml_logger import OutputHandler


def evaluate(model, epoch, sim, eval_episodes=10):
    print("..............................................")
    print(f"Epoch {epoch}. Evaluating scenarios")
    avg_reward = 0.0
    col = 0
    goals = 0
    override_lin = []
    override_ang = []
    override_reward = []
    for _ in range(eval_episodes):
        count = 0

        observation = sim.reset()
        done = False
        while not done and count < 301:
            state, terminal = model.prepare_state(observation)
            action, dev_action = model.get_action(state, False)
            override_lin.append(dev_action[0])
            override_ang.append(dev_action[1])
            observation = sim.step(
                lin_velocity=action[0],
                ang_velocity=action[1],
                override_lin=dev_action[0],
                override_ang=dev_action[1],
            )
            override_reward.append(-observation["reward"]["deviation"])
            avg_reward += (
                observation["reward"]["collision"]
                + observation["reward"]["goal"]
                + observation["reward"]["step"]
                - observation["reward"]["deviation"]
            )
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
    av_ov_lin = statistics.mean(override_lin)
    std_ov_lin = statistics.stdev(override_lin)
    av_ov_ang = statistics.mean(override_ang)
    std_ov_ang = statistics.stdev(override_ang)
    av_dev_reward = statistics.mean(override_reward)

    eval_result = {
        "avg_reward": avg_reward,
        "avg_col": avg_col,
        "avg_goal": avg_goal,
        "av_ov_lin": av_ov_lin,
        "std_ov_lin": std_ov_lin,
        "av_ov_ang": av_ov_ang,
        "std_ov_ang": std_ov_ang,
        "av_dev_reward": av_dev_reward,
    }
    return eval_result


def attach_logging(clearml_logger, trainer, cfg):
    clearml_logger.attach_output_handler(
        trainer,
        tag="base_training",
        event_name=Events.EPOCH_COMPLETED(every=cfg.train_every_n),
        output_transform=lambda _: {
            "base_loss": trainer.state.output["base_loss"],
            "base_avg_Q": trainer.state.output["base_avg_Q"],
            "base_max_Q": trainer.state.output["base_max_Q"],
        },
    )

    clearml_logger.attach_output_handler(
        trainer,
        tag="dev_training",
        event_name=Events.EPOCH_COMPLETED(every=cfg.train_every_n),
        output_transform=lambda _: {
            "dev_loss": trainer.state.output["dev_loss"],
            "dev_avg_Q": trainer.state.output["dev_avg_Q"],
            "dev_max_Q": trainer.state.output["dev_max_Q"],
        },
    )

    clearml_logger.attach(
        trainer,
        log_handler=OutputHandler(
            tag="eval",
            output_transform=lambda _: {
                "avg_reward": trainer.state.eval["avg_reward"],
                "avg_col": trainer.state.eval["avg_col"],
                "avg_goal": trainer.state.eval["avg_goal"],
            },
        ),
        event_name=Events.EPOCH_COMPLETED(every=cfg.episodes_per_epoch),
    )

    clearml_logger.attach(
        trainer,
        log_handler=OutputHandler(
            tag="dev_eval",
            output_transform=lambda _: {
                "av_ov_lin": trainer.state.eval["av_ov_lin"],
                "std_ov_lin": trainer.state.eval["std_ov_lin"],
                "av_ov_ang": trainer.state.eval["av_ov_ang"],
                "std_ov_ang": trainer.state.eval["std_ov_ang"],
                "av_dev_reward": trainer.state.eval["av_dev_reward"],
            },
        ),
        event_name=Events.EPOCH_COMPLETED(every=cfg.episodes_per_epoch),
    )


def upload_metrics(clearml_logger, trainer):
    metrics = {
        "avg_reward": trainer.state.eval["avg_reward"],
        "avg_col": trainer.state.eval["avg_col"],
        "avg_goal": trainer.state.eval["avg_goal"],
    }

    for k, v in metrics.items():
        clearml_logger.report_scalar(
            title="eval",
            series=k,
            value=v,
            iteration=trainer.state.epoch,  # or global step
        )

    metrics = {
        "av_ov_lin": trainer.state.eval["av_ov_lin"],
        "std_ov_lin": trainer.state.eval["std_ov_lin"],
        "av_ov_ang": trainer.state.eval["av_ov_ang"],
        "std_ov_ang": trainer.state.eval["std_ov_ang"],
        "av_dev_reward": trainer.state.eval["av_dev_reward"],
    }

    for k, v in metrics.items():
        clearml_logger.report_scalar(
            title="dev_eval",
            series=k,
            value=v,
            iteration=trainer.state.epoch,  # or global step
        )


def init_checkpoint(trainer, model, cfg):
    to_save = {
        "actor": model.actor,
        "actor_target": model.actor_target,
        "critic": model.critic,
        "critic_target": model.critic_target,
        "actor_optimizer": model.actor_optimizer,
        "critic_optimizer": model.critic_optimizer,
        "dev_actor": model.dev_actor,
        "dev_actor_target": model.dev_actor_target,
        "dev_critic": model.dev_critic,
        "dev_critic_target": model.dev_critic_target,
        "dev_actor_optimizer": model.dev_actor_optimizer,
        "dev_critic_optimizer": model.dev_critic_optimizer,
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
