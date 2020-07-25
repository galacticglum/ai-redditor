'''
SocketIO events.

'''

from flask_socketio import SocketIO, join_room
from ai_redditor_service.extensions import socketio

@socketio.on('join_room', namespace='/app')
def join_room_event_handler(room_id):
    '''
    Sent by clients to join the specified room (by id).    

    '''

    join_room(room_id)
