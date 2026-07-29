from unified_planning.domains.machine_shop import Machine_Shop
from unified_planning.domains.nasa_rover import Nasa_Rover
from unified_planning.domains.stuck_car_1o import Stuck_Car_1o


class NasaRoverNoDeadline(Nasa_Rover):
    """Nasa Rover variant with identical goals and no deadline constraint."""

    def add_goal(self, deadline=None):
        communicated_rock_data, communicated_image_data = self.get_fluents(
            ["communicated_rock_data", "communicated_image_data"]
        )
        objective_list = self.get_objects(
            ["o" + str(i) for i in range(self.object_amount)]
        )
        rock_list = self.get_objects(
            ["x" + str(i) for i in range(self.object_amount * 2)]
        )
        for rock in rock_list:
            self.problem.add_goal(communicated_rock_data(rock))
        for objective in objective_list:
            self.problem.add_goal(communicated_image_data(objective))


class StuckCar1oNoDeadline(Stuck_Car_1o):
    """Stuck Car (1 object) variant with no deadline."""

    def add_goal(self, deadline=None):
        car_out = self.problem.fluent_by_name("car_out")
        self.problem.add_goal(car_out)


class MachineShopNoDeadline(Machine_Shop):
    """Machine Shop variant with same goals and no deadline."""

    def add_goal(self, deadline=None):
        shaped, painted, smooth, polished, free = self.get_fluents(
            ["shaped", "painted", "smooth", "polished", "free"]
        )
        piece_list = self.get_objects(["x" + str(i) for i in range(self.object_amount)])
        machine_list = self.get_objects(
            ["m" + str(i) for i in range(self.object_amount)]
        )

        for i in range(0, self.object_amount):
            self.problem.add_goal(free(machine_list[i]))
            if i % 2 == 0:
                self.problem.add_goal(shaped(piece_list[i]))
                self.problem.add_goal(smooth(piece_list[i]))
            else:
                self.problem.add_goal(painted(piece_list[i]))
                self.problem.add_goal(polished(piece_list[i]))


DOMAIN_FACTORIES = {
    "nasa_rover": NasaRoverNoDeadline,
    "stuck_car_1o": StuckCar1oNoDeadline,
    "machine_shop": MachineShopNoDeadline,
}

