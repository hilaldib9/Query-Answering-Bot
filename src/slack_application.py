import os
import time
from slackclient import SlackClient
from alpha_layer import AlphaLayer

OFFICIAL_DATASET_PATH = '../dataset/dialogue.txt'
SLACK_DATASET_PATH = '../dataset/slack_dialogue.txt'
SPLIT_CHARACTER = '|'
ENTER_MESSAGE = "Arigato Gozaimasu!"
BOT_ID = "U3N0K6TLH"
AT_BOT = "<@" + BOT_ID + ">"

# instantiate Slack & Twilio clients
slack_client = SlackClient('xoxb-124019231697-ZA2MUS39Qky5vdBjT4DZHHI1')



alpha = AlphaLayer(OFFICIAL_DATASET_PATH)
    
def restart_program():
    """Restarts the current program.
    Note: this function does not return. Any cleanup action (like
    saving data) must be done before calling this function."""
    python = sys.executable
    os.execl(python, python, * sys.argv)
    
def handle_command(command, channel):
    
    """
        Receives commands directed at the bot and determines if they
        are valid commands. If so, then acts on the commands.
    """
    
    
    message = command.split('|', 1)[1]
    command = command.split('|', 1)[0]

    
    if command == 'train':
        #TODO: ERROR HANDLING
        """
        classifier = message.split('|', 1)[0]
        sentence = message.split('|', 1)[1]
        
        responses = ["Thank you for training me!\nClassifier: ",
                     classifier + "\nSentence: ",
                     sentence]
        
        
        slack_client.api_call("chat.postMessage", channel=channel,
                          text=''.join(responses), as_user=True)
        
        alpha.add_line(message)
        alpha.load()
        
        slack_client.api_call("chat.postMessage", channel=channel,
                          text="Done re-training", as_user=True)   
        """
        
        
        slack_client.api_call("chat.postMessage", channel=channel,
                          text="Training has been suspended.", as_user=True)           
    elif command == 'test':
        
        features = alpha.fe.extract_word_features_phrase(message)
        vector = alpha.fe.prediction_vector(features)
        results = alpha.mb.get_predictions(vector)
        response = ''.join(results)
        
        slack_client.api_call("chat.postMessage", channel=channel,
                          text=response, as_user=True)
    elif command == "reset":
        slack_client.api_call("chat.postMessage", channel=channel,
                          text="Resetting! Time to learn :).", as_user=True)  
        alpha.load()
        slack_client.api_call("chat.postMessage", channel=channel,
                          text="Finished reset.", as_user=True)          
    else:
        response = "Thats an invalid command. Train me!"
        slack_client.api_call("chat.postMessage", channel=channel,
                          text=response, as_user=True)        


def parse_slack_output(slack_rtm_output):
    
    '''
        The Slack Real Time Messaging API is an events firehose.
        this parsing function returns None unless a message is
        directed at the Bot, based on its ID.
    '''
    
    output_list = slack_rtm_output
    if output_list and len(output_list) > 0:
        for output in output_list:
            if output and 'text' in output and AT_BOT in output['text']:
                # return text after the @ mention, whitespace removed
                return output['text'].split(AT_BOT)[1].strip().lower(), \
                       output['channel']
    return None, None


if __name__ == "__main__":
    
    READ_WEBSOCKET_DELAY = 1
    
    if slack_client.rtm_connect():
        slack_client.api_call("chat.postMessage", channel='we_r_the_bot_trainers',
                              text=ENTER_MESSAGE, as_user=True)
        
        print("AoABot connected and running!")
        while True:
            command, channel = parse_slack_output(slack_client.rtm_read())
            if command and channel:
                handle_command(command, channel)
            time.sleep(READ_WEBSOCKET_DELAY)
    else:
        print("Connection failed. Invalid Slack token or bot ID?")