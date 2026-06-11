from models.VOT.tracking.actions import ActionConfig, ActionHistory, apply_action
from models.VOT.tracking.actions import ORIGINAL_ADNET_ACTIONS, DRONE_ADNET_ACTIONS


def test_left_moves_center_left():
    box = [100, 100, 50, 40]
    new_box = apply_action(box, "left", ActionConfig(alpha=0.1))
    assert new_box == [95, 100, 50, 40]


def test_up_uses_height_not_width():
    box = [100, 100, 50, 40]
    new_box = apply_action(box, "up", ActionConfig(alpha=0.1))
    assert new_box == [100, 96, 50, 40]


def test_stop_does_not_change_box():
    box = [100, 100, 50, 40]
    assert apply_action(box, "stop") == box


def test_history_vector_size_original():
    h = ActionHistory(ORIGINAL_ADNET_ACTIONS)
    assert h.as_vector().shape == (110,)


if __name__ == "__main__":
    test_left_moves_center_left()
    test_up_uses_height_not_width()
    test_stop_does_not_change_box()
    test_history_vector_size_original()
    print("All tests passed!")