import numpy as np
from ignite.engine import Events
from ignite.handlers.clearml_logger import OutputHandler

def evaluate(model, epoch, sim, eval_episodes=10):
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
            a = compute_action(observation["distance"], observation["sin"], observation["cos"])
            state, terminal = model.prepare_state(observation, a)
            action = model.get_action(np.array(state), False)
            observation = sim.step(
                lin_velocity=a[0],
                ang_velocity=a[1],
                override_lin=action[0],
                override_ang=action[1],
                switch=False,
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
        tag="training",
        event_name=Events.EPOCH_COMPLETED(every=cfg.train_every_n),
        output_transform=lambda out: out,
    )
    clearml_logger.attach(
        trainer,
        log_handler=OutputHandler(
            tag="eval",
            metric_names=["avg_reward", "avg_col", "avg_goal"],
            output_transform=lambda _: {
                "avg_reward": trainer.state.eval["avg_reward"],
                "avg_col": trainer.state.eval["avg_col"],
                "avg_goal": trainer.state.eval["avg_goal"],
            }
        ),
        event_name=Events.EPOCH_COMPLETED(every=cfg.episodes_per_epoch),
    )