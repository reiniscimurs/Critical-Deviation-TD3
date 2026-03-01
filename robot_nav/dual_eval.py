from ignite.engine import Engine
from utils.utils import setup_clearm
import hydra
from omegaconf import DictConfig
from utils.dual_utils import attach_logging, evaluate, init_checkpoint, upload_metrics


@hydra.main(version_base=None, config_path="configs", config_name="dual_eval_config")
def main(cfg: DictConfig):
    """Main training function"""

    clearml_logger = setup_clearm(cfg.clearml)
    train_model = hydra.utils.instantiate(cfg.base_model)

    sim = hydra.utils.instantiate(cfg.sim)

    def run_step(engine, timestep):
        pass

    trainer = Engine(run_step)
    init_checkpoint(trainer, train_model, cfg)
    trainer.state.eval = evaluate(
        train_model, trainer.state.epoch, sim, eval_episodes=cfg.nr_eval_episodes
    )
    upload_metrics(clearml_logger, trainer)

    clearml_logger.close()


if __name__ == "__main__":
    main()
