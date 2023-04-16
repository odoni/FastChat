from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTListIndex,GPTSimpleVectorIndex, PromptHelper
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LLMPredictor, ServiceContext
import torch
from langchain.llms.base import LLM
from transformers import pipeline
import logging

logging.getLogger().setLevel(logging.CRITICAL)

class customLLM(LLM):
    model_name = "google/flan-t5-large"
    pipeline = pipeline("text2text-generation", model=model_name, device=0, model_kwargs={"torch_dtype":torch.bfloat16})

    def _call(self, prompt, stop=None):
        return self.pipeline(prompt, max_length=9999)[0]["generated_text"]
 
    def _identifying_params(self):
        return {"name_of_model": self.model_name}

    def _llm_type(self):
        return "custom"


llm_predictor = LLMPredictor(llm=customLLM())

hfemb = HuggingFaceEmbeddings()
embed_model = LangchainEmbedding(hfemb)

text1 = """The use of NLP in the realm of financial technology is broad and complex, with applications
ranging from sentiment analysis and named entity recognition to question answering. Large
Language Models (LLMs) have been shown to be effective on a variety of tasks; however, no
LLM specialized for the financial domain has been reported in literature. In this work, we
present BloombergGPT, a 50 billion parameter language model that is trained on a wide
range of financial data. We construct a 363 billion token dataset based on Bloomberg’s
extensive data sources, perhaps the largest domain-specific dataset yet, augmented with 345
billion tokens from general purpose datasets. We validate BloombergGPT on standard
LLM benchmarks, open financial benchmarks, and a suite of internal benchmarks that most
accurately reflect our intended usage. Our mixed dataset training leads to a model that
outperforms existing models on financial tasks by significant margins without sacrificing
performance on general LLM benchmarks. Additionally, we explain our modeling choices,
training process, and evaluation methodology. As a next step, we plan to release training
logs (Chronicles) detailing our experience in training BloombergGPT.
Snap Inc., the creator of Snapchat, introduced My AI for Snapchat+ this week. The experimental feature is running on ChatGPT API. My AI offers Snapchatters a friendly, customizable chatbot at their fingertips that offers recommendations, and can even write a haiku for friends in seconds. Snapchat, where communication and messaging is a daily behavior, has 750 million monthly Snapchatters.


Quizlet Q-Chat, UI screenshot

Play video
Quizlet Q-Chat

Quizlet is a global learning platform with more than 60 million students using it to study, practice and master whatever they’re learning. Quizlet has worked with OpenAI for the last three years, leveraging GPT-3 across multiple use cases, including vocabulary learning and practice tests. With the launch of ChatGPT API, Quizlet is introducing Q-Chat, a fully-adaptive AI tutor that engages students with adaptive questions based on relevant study materials delivered through a fun chat experience.


Instacart’s Ask Instacart, UI screenshot
Instacart’s Ask Instacart

Instacart is augmenting the Instacart app to enable customers to ask about food and get inspirational, shoppable answers. This uses ChatGPT alongside Instacart’s own AI and product data from their 75,000+ retail partner store locations to help customers discover ideas for open-ended shopping goals, such as “How do I make great fish tacos?” or “What’s a healthy lunch for my kids?” Instacart plans to launch “Ask Instacart” later this year.



Play video
Shopify’s Shop app

Shop, Shopify’s consumer app, is used by 100 million shoppers to find and engage with the products and brands they love. ChatGPT API is used to power Shop’s new shopping assistant. When shoppers search for products, the shopping assistant makes personalized recommendations based on their requests. Shop’s new AI-powered shopping assistant will streamline in-app shopping by scanning millions of products to quickly find what buyers are looking for—or help them discover something new.


The Speak App, UI screenshot

Play video
The Speak app

Speak is an AI-powered language learning app focused on building the best path to spoken fluency. They’re the fastest-growing English app in South Korea, and are already using the Whisper API to power a new AI speaking companion product, and rapidly bring it to the rest of the globe. Whisper’s human-level accuracy for language learners of every level unlocks true open-ended conversational practice and highly accurate feedback.

ChatGPT API
Model: The ChatGPT model family we are releasing today, gpt-3.5-turbo, is the same model used in the ChatGPT product. It is priced at 
OPENAI_API_KEY"
  -H "Content-Type: application/json"
  -d '{
  "model": "gpt-3.5-turbo",
  "messages": [{"role": "user", "content": "What is the OpenAI mission?"}]
}'
To learn more about the ChatGPT API, visit our Chat guide.

ChatGPT upgrades
We are constantly improving our ChatGPT models, and want to make these enhancements available to developers as well. Developers who use the gpt-3.5-turbo model will always get our recommended stable model, while still having the flexibility to opt for a specific model version. For example, today we’re releasing gpt-3.5-turbo-0301, which will be supported through at least June 1st, and we’ll update gpt-3.5-turbo to a new stable release in April. The models page will provide switchover updates.

Dedicated instances
We are also now offering dedicated instances for users who want deeper control over the specific model version and system performance. By default, requests are run on compute infrastructure shared with other users, who pay per request. Our API runs on Azure, and with dedicated instances, developers will pay by time period for an allocation of compute infrastructure that’s reserved for serving their requests.

Developers get full control over the instance’s load (higher load improves throughput but makes each request slower), the option to enable features such as longer context limits, and the ability to pin the model snapshot.

Dedicated instances can make economic sense for developers running beyond ~450M tokens per day. Additionally, it enables directly optimizing a developer’s workload against hardware performance, which can dramatically reduce costs relative to shared infrastructure. For dedicated instance inquiries, contact us.

Whisper API
Whisper, the speech-to-text model we open-sourced in September 2022, has received immense praise from the developer community but can also be hard to run. We’ve now made the large-v2 model available through our API, which gives convenient on-demand access priced at 
OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F model="whisper-1" \
  -F file="@/path/to/file/openai.mp3"
To learn more about the Whisper API, visit our Speech to Text guide.

Developer focus
Over the past six months, we’ve been collecting feedback from our API customers to understand how we can better serve them. We’ve made concrete changes, such as:

Data submitted through the API is no longer used for service improvements (including model training) unless the organization opts in
Implementing a default 30-day data retention policy for API users, with options for stricter retention depending on user needs.
Removing our pre-launch review (unlocked by improving our automated monitoring)
Improving developer documentation
Simplifying our Terms of Service and Usage Policies, including terms around data ownership: users own the input and output of the models.
For the past two months our uptime has not met our own expectations nor that of our users. Our engineering team’s top priority is now stability of production use cases—we know that ensuring AI benefits all of humanity requires being a reliable service provider. Please hold us accountable for improved uptime over the upcoming months!

We believe that AI can provide incredible opportunities and economic empowerment to everyone, and the best way to achieve that is to allow everyone to build with it. We hope that the changes we announced today will lead to numerous applications that everyone can benefit from. Start building next-generation apps powered by ChatGPT & Whisper."""
 
 #documents = SimpleDirectoryReader('data').load_data()

from llama_index import Document

text_list = [text1]

documents = [Document(t) for t in text_list]

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model)
index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)

response = index.query( "What is the age of universe?")

print(response.response)