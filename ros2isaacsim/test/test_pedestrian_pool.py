from isaac_utils.pedestrian_pool import parking_pose_for_name


def test_parking_pose_is_stable_for_same_character():
    first = parking_pose_for_name("/World/Characters/toilet_agent_01")
    second = parking_pose_for_name("/World/Characters/toilet_agent_01")

    assert first == second


def test_benchmark_agents_receive_separated_parking_slots():
    poses = {
        tuple(parking_pose_for_name(f"/World/Characters/toilet_agent_{index:02d}"))
        for index in range(1, 9)
    }

    assert len(poses) == 8
    assert all(pose[0] >= 1000.0 and pose[1] >= 1000.0 for pose in poses)
