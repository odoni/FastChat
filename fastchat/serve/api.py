import argparse
import re
from fastchat.serve.inference import api_chat, load_api_model
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

class ApiModel:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer


@app.route("/", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        # get the user's message from the form
        json_data = request.get_json()
        prompt = clean_prompt(json_data.get("message"))
        print(prompt)
        output = api_chat(ApiModel.model, ApiModel.tokenizer, args.model_name, args.device, args.conv_template, args.temperature, args.max_new_tokens, args.debug, prompt)
        return jsonify({"text": output})
    else:
        # render the chat web interface
        return render_template("chat.html")

def clean_prompt(prompt: str):
    new_prompt =  prompt.replace("Use the following portion of a long document to see if any of the text is relevant to answer the question.", "")
    new_prompt =  new_prompt.replace("Return any relevant text verbatim.", "")
    new_prompt =  new_prompt.replace("#", "")
    new_prompt =  new_prompt.replace("**", "")
    new_prompt = "### input\r\n" + new_prompt
    return new_prompt

def main(args):
    ApiModel.model, ApiModel.tokenizer = load_api_model(args.model_name, args.device, args.num_gpus, args.load_8bit, args.debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default="cuda")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--load-8bit", action="store_true",
        help="Use 8-bit quantization.")
    parser.add_argument("--conv-template", type=str, default="v1",
        help="Conversation prompt template.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--style", type=str, default="simple",
                        choices=["simple", "rich"], help="Display style.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
    app.run(debug=False, port=7860, host='0.0.0.0')