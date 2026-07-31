from pedestrian.simulator.logic.people_manager import PeopleManager
from isaac_utils.managers.door_manager import door_manager
from isaac_utils.pedestrian_pool import park_all_people

from isaacsim_msgs.srv import DeletePrim

from .utils import safe


@safe
def characters_delete(request, response):
    park_all_people(PeopleManager.get_people_manager())
    door_manager.reset()
    response.ret = True
    return response


def delete_all_characters(controller):
    service = controller.create_service(
        srv_type=DeletePrim,
        srv_name='isaac/delete_all_pedestrians',
        callback=characters_delete
    )
    return service
