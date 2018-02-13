import apiai
import json

class AiController:
    """This class does helps with the methodes to control my ai."""

    CLIENT_ACCESS_TOKEN = 'd031bde1cf5b4c26b60326fe8c395ef7'
    ai = apiai.ApiAI(CLIENT_ACCESS_TOKEN)

    def send_text_message(self, sender_id, text_message):
        """This methode sends a text message and returns the response"""
        request = self.ai.text_request()
        request.session_id = sender_id
        request.query = text_message

        response = json.loads(request.getresponse().read().decode('utf-8'))
        return response

    def get_text_from_response(self, response):
        """Get text."""
        text = response['result']['fulfillment']['speech']
        return text
