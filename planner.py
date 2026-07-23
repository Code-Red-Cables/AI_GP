class Planner:
    """Base planner: commands zero velocity (hover in place)."""
    name = 'hover'

    def compute_target(self, shared_data):
        shared_data['planner_mode'] = self.name
        return {'vn': 0.0, 've': 0.0, 'vd': 0.0, 'yaw_rate': 0.0}
