# Should accept cmd line parameters such as: hostname, port, queue name for recieving cas, and queue name for
# sending cas

# These are the lines that ignore the typesystem errors
import warnings

from main_folder.pbj_receiver_v2 import PbjReceiver
from main_folder.pbj_sender_v2 import PBJSender
from main_folder.pbj_util import TypeSystemAccessor, DEFAULT_HOST, DEFAULT_PORT
from main_folder.pipeline import Pipeline

# import pbj_receiver
# from pbj_sender import *
from example_cnlpt_pipeline import ExampleCnlptPipeline

warnings.filterwarnings("ignore")


def main():
    hostname = DEFAULT_HOST
    port = DEFAULT_PORT
    queue_receive_cas = 'test/JavaToPython'
    queue_send_cas = 'test/PythonToJava'

    print(hostname)
    print(port)
    print(queue_receive_cas)
    print(queue_send_cas)
    # start the receiver
    # receiver(queue_receive_cas)
    # once a cas has been received we should start a sender to send the cas
    # start the sender
    pipeline = Pipeline()
    pipeline.add(ExampleCnlptPipeline(TypeSystemAccessor().get_type_system()))
    pipeline.add(PBJSender(queue_send_cas))

    PbjReceiver(pipeline, queue_receive_cas)


main()
