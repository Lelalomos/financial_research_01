from pathlib import Path
import re


REPO_ROOT = Path(__file__).parent.parent


def test_docker_services_allow_mlflow_file_store():
    contents = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    for service_name in ["crnn-prediction", "trainer"]:
        service_start = contents.index(f"  {service_name}:")
        next_service_match = re.search(r"\n  [A-Za-z0-9_-]+:\n", contents[service_start + 1 :])
        next_service = -1 if next_service_match is None else service_start + 1 + next_service_match.start()
        service_block = contents[service_start:] if next_service == -1 else contents[service_start:next_service]

        assert "- MLFLOW_ALLOW_FILE_STORE=true" in service_block
