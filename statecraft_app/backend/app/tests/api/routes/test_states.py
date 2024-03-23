from app.core.config import settings
from app.tests.utils.state import create_random_state
from fastapi.testclient import TestClient
from sqlmodel import Session


def test_create_state(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    data = {"title": "Foo", "description": "Fighters"}
    response = client.post(
        f"{settings.API_V1_STR}/states/",
        headers=superuser_token_headers,
        json=data,
    )
    assert response.status_code == 200
    content = response.json()
    assert content["title"] == data["title"]
    assert content["description"] == data["description"]
    assert "id" in content
    assert "owner_id" in content


def test_read_state(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    state = create_random_state(db)
    response = client.get(
        f"{settings.API_V1_STR}/states/{state.id}",
        headers=superuser_token_headers,
    )
    assert response.status_code == 200
    content = response.json()
    assert content["title"] == state.title
    assert content["description"] == state.description
    assert content["id"] == state.id
    assert content["owner_id"] == state.owner_id


def test_read_state_not_found(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    response = client.get(
        f"{settings.API_V1_STR}/states/999",
        headers=superuser_token_headers,
    )
    assert response.status_code == 404
    content = response.json()
    assert content["detail"] == "State 999 not found"


def test_read_state_not_enough_permissions(
    client: TestClient, normal_user_token_headers: dict[str, str], db: Session
) -> None:
    state = create_random_state(db)
    response = client.get(
        f"{settings.API_V1_STR}/states/{state.id}",
        headers=normal_user_token_headers,
    )
    assert response.status_code == 400
    content = response.json()
    assert content["detail"] == "Not enough permissions"


def test_read_states(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    create_random_state(db)
    create_random_state(db)
    response = client.get(
        f"{settings.API_V1_STR}/states/",
        headers=superuser_token_headers,
    )
    assert response.status_code == 200
    content = response.json()
    assert len(content["data"]) >= 2


def test_update_state(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    state = create_random_state(db)
    data = {"title": "Updated title", "description": "Updated description"}
    response = client.put(
        f"{settings.API_V1_STR}/states/{state.id}",
        headers=superuser_token_headers,
        json=data,
    )
    assert response.status_code == 200
    content = response.json()
    assert content["title"] == data["title"]
    assert content["description"] == data["description"]
    assert content["id"] == state.id
    assert content["owner_id"] == state.owner_id


def test_update_state_not_found(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    data = {"title": "Updated title", "description": "Updated description"}
    response = client.put(
        f"{settings.API_V1_STR}/states/999",
        headers=superuser_token_headers,
        json=data,
    )
    assert response.status_code == 404
    content = response.json()
    assert content["detail"] == "State not found"


def test_update_state_not_enough_permissions(
    client: TestClient, normal_user_token_headers: dict[str, str], db: Session
) -> None:
    state = create_random_state(db)
    data = {"title": "Updated title", "description": "Updated description"}
    response = client.put(
        f"{settings.API_V1_STR}/states/{state.id}",
        headers=normal_user_token_headers,
        json=data,
    )
    assert response.status_code == 400
    content = response.json()
    assert content["detail"] == "Not enough permissions"


def test_delete_state(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    state = create_random_state(db)
    response = client.delete(
        f"{settings.API_V1_STR}/states/{state.id}",
        headers=superuser_token_headers,
    )
    assert response.status_code == 200
    content = response.json()
    assert content["message"] == f"State {state.id} deleted successfully"


def test_delete_state_not_found(
    client: TestClient, superuser_token_headers: dict[str, str], db: Session
) -> None:
    response = client.delete(
        f"{settings.API_V1_STR}/states/999",
        headers=superuser_token_headers,
    )
    assert response.status_code == 404
    content = response.json()
    assert content["detail"] == "State 999 not found"


def test_delete_state_not_enough_permissions(
    client: TestClient, normal_user_token_headers: dict[str, str], db: Session
) -> None:
    state = create_random_state(db)
    response = client.delete(
        f"{settings.API_V1_STR}/states/{state.id}",
        headers=normal_user_token_headers,
    )
    assert response.status_code == 400
    content = response.json()
    assert content["detail"] == "Not enough permissions"
