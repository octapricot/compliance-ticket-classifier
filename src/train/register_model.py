"""
Promote a trained model artifact into the W&B Model Registry.
"""
import wandb
from dotenv import load_dotenv

load_dotenv()

ENTITY = "k-dubas-set-university"
PROJECT = "compliance-ticket-classifier"

# The artifact we want to promote (source).
SOURCE_ARTIFACT = f"{ENTITY}/{PROJECT}/distilbert-compliance:v0"

# The registry collection to promote it into (target).
# Path format: wandb-registry-<registry>/<collection>
TARGET_COLLECTION = "wandb-registry-model/compliance-classifier"


def main():
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name="register-distilbert",
        job_type="model-registry",
    )

    artifact = run.use_artifact(SOURCE_ARTIFACT, type="model")

    run.link_artifact(
        artifact=artifact,
        target_path=TARGET_COLLECTION,
        aliases=["production"],   # <- the alias the serving code will pull
    )

    print(f"Linked {SOURCE_ARTIFACT}")
    print(f"    -> {TARGET_COLLECTION} (alias: production)")
    run.finish()


if __name__ == "__main__":
    main()