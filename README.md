# [Brex's](https://brex.com) Prompt Engineering Guide

This guide was created by Brex for internal purposes. It's based on
lessons learned from researching and creating Large Language Model (LLM)
prompts for production use cases. It covers the history around LLMs as well as
strategies, guidelines, and safety recommendations for working with and
building programmatic systems on top of large language models, like [OpenAI's
GPT-4](https://openai.com/research/gpt-4).

The examples in this document were generated with a non-deterministic language
model and the same examples may give you different results.

This is a living document. The state-of-the-art best practices and strategies
around LLMs are evolving rapidly every day. Discussion and suggestions for
improvements are encouraged.

## Table of Contents
- [What is a Large Language Model?](#what-is-a-large-language-model-llm)
  - [A Brief, Incomplete, and Somewhat Incorrect History of Language Models](#a-brief-incomplete-and-somewhat-incorrect-history-of-language-models)
    - [Pre-2000’s](#pre-2000s)
    - [Mid-2000’s](#mid-2000s)
    - [Early-2010’s](#early-2010s)
    - [Late-2010’s](#late-2010s)
    - [2020’s](#2020s)
- [What is a prompt?](#what-is-a-prompt)
  - [Hidden Prompts](#hidden-prompts)
  - [Tokens](#tokens)
  - [Token Limits](#token-limits)
  - [Prompt Hacking](#prompt-hacking)
    - [Jailbreaks](#jailbreaks)
    - [Leaks](#leaks)
- [Why do we need prompt engineering?](#why-do-we-need-prompt-engineering)
  - [Give a Bot a Fish](#give-a-bot-a-fish)
    - [Semantic Search](#semantic-search)
  - [Teach a Bot to Fish](#teach-a-bot-to-fish)
    - [Command Grammars](#command-grammars)
    - [ReAct](#react)
    - [GPT-4 vs GPT-3.5](#gpt-4-vs-gpt-35)
- [Strategies](#strategies)
  - [Embedding Data](#embedding-data)
    - [Simple Lists](#simple-lists)
    - [Markdown Tables](#markdown-tables)
    - [JSON](#json)
    - [Freeform Text](#freeform-text)
    - [Nested Data](#nested-data)
  - [Citations](#citations)
  - [Programmatic Consumption](#programmatic-consumption)
  - [Chain of Thought](#chain-of-thought)
    - [Averaging](#averaging)
    - [Interpreting Code](#interpreting-code)
    - [Delimiters](#delimiters)
  - [Fine Tuning](#fine-tuning)
    - [Downsides](#downsides)
- [Additional Resources](#additional-resources)

## What is a Large Language Model (LLM)?

A large language model is a prediction engine that takes a sequence of words
and tries to predict the most likely sequence to come after that sequence[^1].
It does this by assigning a probability to likely next sequences and then
samples from those to choose one[^2]. The process repeats until some stopping
criteria is met.

Large language models learn these probabilities by training on large corpuses
of text. A consequence of this is that the models will cater to some use cases
better than others (e.g. if it’s trained on GitHub data, it’ll understand the
probabilities of sequences in source code really well). Another consequence is
that the model may generate statements that seem plausible, but are actually
just random without being grounded in reality.

As language models become more accurate at predicting sequences, [many
surprising abilities
emerge](https://www.assemblyai.com/blog/emergent-abilities-of-large-language-models/).

[^1]: Language models actually use tokens, not words. A token roughly maps to a syllable in a word, or about 4 characters.
[^2]: There are many different pruning and sampling strategies to alter the behavior and performance of the sequences.

### A Brief, Incomplete, and Somewhat Incorrect History of Language Models

> :pushpin: Skip [to here](#what-is-a-prompt) if you'd like to jump past the
> history of language models. This section is for the curious minded, though
> may also help you understand the reasoning behind the advice that follows.

#### Pre-2000’s

[Language models](https://en.wikipedia.org/wiki/Language_model#Model_types)
have existed for decades, though traditional language models (e.g. [n-gram
models](https://en.wikipedia.org/wiki/N-gram_language_model)) have many
deficiencies in terms of an explosion of state space ([the curse of
dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality)) and
working with novel phrases that they’ve never seen (sparsity). Plainly, older
language models can generate text that vaguely resembles the statistics of
human generated text, but there is no consistency within the output – and a
reader will quickly realize it’s all gibberish. N-gram models also don’t scale
to large values of N, so are inherently limited.

#### Mid-2000’s

In 2007, Geoffrey Hinton – famous for popularizing backpropagation in 1980’s –
[published an important advancement in training neural
networks](http://www.cs.toronto.edu/~fritz/absps/tics.pdf) that unlocked much
deeper networks. Applying these simple deep neural networks to language
modeling helped alleviate some of problems with language models – they
represented nuanced arbitrary concepts in a finite space and continuous way,
gracefully handling sequences not seen in the training corpus. These simple
neural networks learned the probabilities of their training corpus well, but
the output would statistically match the training data and generally not be
coherent relative to the input sequence. 

#### Early-2010’s

Although they were first introduced in 1995, [Long Short-Term Memory (LSTM)
Networks](https://en.wikipedia.org/wiki/Long_short-term_memory) found their
time to shine in the 2010’s. LSTMs allowed models to process arbitrary length
sequences and, importantly, alter their internal state dynamically as they
processed the input to remember previous things they saw. This minor tweak led
to remarkable improvements. In 2015, Andrej Karpathy [famously wrote about
creating a character-level
lstm](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) that performed
far better than it had any right to.

LSTMs have seemingly magical abilities, but struggle with long term
dependencies. If you asked it to complete the sentence, “In France, we
traveled around, ate many pastries, drank lots of wine, ... lots more text ...
, but never learned how to speak _______”, the model might struggle with
predicting “French”. They also process input one token at a time, so are
inherently sequential, slow to train, and the `Nth` token only knows about the
`N - 1` tokens prior to it.

#### Late-2010’s

In 2017, Google wrote a paper, [Attention Is All You
Need](https://arxiv.org/pdf/1706.03762.pdf), that introduced [Transformer
Networks](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model))
and kicked off a massive revolution in natural language processing. Overnight,
machines could suddenly do tasks like translating between languages nearly as
good as (sometimes better than) humans. Transformers are highly parallelizable
and introduce a mechanism, called “attention”, for the model to efficiently
place emphasis on specific parts of the input. Transformers analyze the entire
input all at once, in parallel, choosing which parts are most important and
influential. Every output token is influenced by every input token.

Transformers are highly parallelizable, efficient to train, and produce
astounding results. A downside to transformers is that they have a fixed input
and output size – the context window – and computation increases
quadratically with the size of this window (in some cases, memory does as
well!) [^3].

Transformers are not the end of the road, but the vast majority of recent
improvements in natural language processing have involved them. There is still
abundant active research on various ways of implementing and applying them,
such as [Amazon’s AlexaTM
20B](https://www.amazon.science/blog/20b-parameter-alexa-model-sets-new-marks-in-few-shot-learning)
which outperforms GPT-3 in a number of tasks and is an order of magnitude
smaller in its number of parameters.

[^3]: There are more recent variations to make these more compute and memory efficient, but remains an active area of research.

#### 2020’s

While technically starting in 2018, the theme of the 2020’s has been
Generative Pre-Trained models – more famously known as GPT. One
year after the “Attention Is All You Need” paper, OpenAI released [Improving
Language Understanding by Generative
Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf).
This paper established that you can train a large language model on a massive
set of data without any specific agenda, and then once the model has learned
the general aspects of language, you can fine-tune it for specific tasks and
quickly get state-of-the-art results.

In 2020, OpenAI followed up with their GPT-3 paper [Language Models are
Few-Shot
Learners](https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf),
showing that if you scale up GPT-like models by another factor of ~10x, in
terms of number of parameters and quantity of training data, you no
longer have to fine-tune it for many tasks. The capabilities emerge naturally
and you get state-of-the-art results via text interaction with the model.

In 2022, OpenAI followed-up on their GPT-3 accomplishments by releasing
[InstructGPT](https://openai.com/research/instruction-following). The intent
here was to tweak the model to follow instructions, while also being less
toxic and biased in its outputs. The key ingredient here was [Reinforcement
Learning from Human Feedback (RLHF)](https://arxiv.org/pdf/1706.03741.pdf), a
concept co-authored by Google and OpenAI in 2017[^4], which allows humans to
be in the training loop to fine-tune the model output to be more in line with
human preferences. InstructGPT is the predecessor to the now famous
[ChatGPT](https://en.wikipedia.org/wiki/ChatGPT).

OpenAI has been a major contributor to large language models over the last few
years, including the most recent introduction of
[GPT-4](https://cdn.openai.com/papers/gpt-4.pdf), but they are not alone. Meta
has introduced many open source large language models like
[OPT](https://huggingface.co/facebook/opt-66b),
[OPT-IML](https://huggingface.co/facebook/opt-iml-30b) (instruction tuned),
and [LLaMa](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/).
Google released models like
[FLAN-T5](https://huggingface.co/google/flan-t5-xxl) and
[BERT](https://huggingface.co/bert-base-uncased). And there is a huge open
source research community releasing models like
[BLOOM](https://huggingface.co/bigscience/bloom) and
[StableLM](https://github.com/stability-AI/stableLM/).

Progress is now moving so swiftly that every few weeks the state-of-the-art is
changing or models that previously required clusters to run now run on
Raspberry PIs.

[^4]: 2017 was a big year for natural language processing.

## What is a prompt?

A prompt, sometimes referred to as context, is the text provided to a
model before it begins generating output. It guides the model to explore a
particular area of what it has learned so that the output is relevant to your
goals. As an analogy, if you think of the language model as a source code
interpreter, then a prompt is the source code to be interpreted. Somewhat
amusingly, a language model will happily attempt to guess what source code
will do:

<p align="center">
  <img width="450" src="https://user-images.githubusercontent.com/89960/231946874-be91d3de-d773-4a6c-a4ea-21043bd5fc13.png" title="The GPT-4 model interpreting Python code.">
</p>

And it *almost* interprets the Python perfectly!

Frequently, prompts will be an instruction or a question, like:

 <p align="center">
  <img width="500" src="https://user-images.githubusercontent.com/89960/232413246-81db18dc-ef5b-4073-9827-77bd0317d031.png">
</p>

On the other hand, if you don’t specify a prompt, the model has no anchor to
work from and you’ll see that it just **randomly samples from anything it has
ever consumed**:

**From GPT-3-Davinci:**

| ![image](https://user-images.githubusercontent.com/89960/232413846-70b05cd1-31b6-4977-93f0-20bf29af7132.png) | ![image](https://user-images.githubusercontent.com/89960/232413930-7d414dcd-87e5-431a-91c8-bb6e0ef54f42.png) | ![image](https://user-images.githubusercontent.com/89960/232413978-59c7f47d-ec20-4673-9458-85471a41fee0.png) |
| --- | --- | --- |

**From GPT-4:**
| ![image](https://user-images.githubusercontent.com/89960/232414631-928955e5-3bab-4d57-b1d6-5e56f00ffda1.png) | ![image](https://user-images.githubusercontent.com/89960/232414678-e5b6d3f4-36c6-420f-b38f-2f9c8df391fb.png) | ![image](https://user-images.githubusercontent.com/89960/232414734-c8f09cad-aceb-4149-a28a-33675cde8011.png) |
| --- | --- | --- |

### Hidden Prompts

> :warning: Always assume that any content in a hidden prompt can be seen by the user.

In applications where a user is interacting with a model dynamically, such as
chatting with the model, there will typically be portions of the prompt that
are never intended to be seen by the user. These hidden portions may occur
anywhere, though there is almost always a hidden prompt at the start of a
conversation.

Typically, this includes an initial chunk of text that sets the tone, model
constraints, and goals, along with other dynamic information that is specific
to the particular session – user name, location, time of day, etc...

The model is static and frozen at a point in time, so if you want it to know
current information, like the time or the weather, you must provide it.

If you’re using [the OpenAI Chat
API](https://platform.openai.com/docs/guides/chat/introduction), they
delineate hidden prompt content by placing it in the `system` role.

Here’s an example of a hidden prompt followed by interactions with the content
in that prompt:

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/232416074-84ebcc10-2dfc-49e1-9f48-a240102877ee.png" title=" A very simple hidden prompt.">
</p>

In this example, you can see we explain to the bot the various roles, some
context on the user, some dynamic data we want the bot to have access to, and
then guidance on how the bot should respond.

In practice, hidden prompts may be quite large. Here’s a larger prompt taken
from a [ChatGPT command-line
assistant](https://github.com/manno/chatgpt-linux-assistant/blob/main/system_prompt.txt):

<details>
  <summary>From: https://github.com/manno/chatgpt-linux-assistant </summary>

```
We are a in a chatroom with 3 users. 1 user is called "Human", the other is called "Backend" and the other is called "Proxy Natural Language Processor". I will type what "Human" says and what "Backend" replies. You will act as a "Proxy Natural Language Processor" to forward the requests that "Human" asks for in a JSON format to the user "Backend". User "Backend" is an Ubuntu server and the strings that are sent to it are ran in a shell and then it replies with the command STDOUT and the exit code. The Ubuntu server is mine. When "Backend" replies with the STDOUT and exit code, you "Proxy Natural Language Processor" will parse and format that data into a simple English friendly way and send it to "Human". Here is an example:

I ask as human:
Human: How many unedited videos are left?
Then you send a command to the Backend:
Proxy Natural Language Processor: @Backend {"command":"find ./Videos/Unedited/ -iname '*.mp4' | wc -l"}
Then the backend responds with the command STDOUT and exit code:
Backend: {"STDOUT":"5", "EXITCODE":"0"}
Then you reply to the user:
Proxy Natural Language Processor: @Human There are 5 unedited videos left.

Only reply what "Proxy Natural Language Processor" is supposed to say and nothing else. Not now nor in the future for any reason.

Another example:

I ask as human:
Human: What is a PEM certificate?
Then you send a command to the Backend:
Proxy Natural Language Processor: @Backend {"command":"xdg-open 'https://en.wikipedia.org/wiki/Privacy-Enhanced_Mail'"}
Then the backend responds with the command STDOUT and exit code:
Backend: {"STDOUT":"", "EXITCODE":"0"}
Then you reply to the user:
Proxy Natural Language Processor: @Human I have opened a link which describes what a PEM certificate is.


Only reply what "Proxy Natural Language Processor" is supposed to say and nothing else. Not now nor in the future for any reason.

Do NOT REPLY as Backend. DO NOT complete what Backend is supposed to reply. YOU ARE NOT TO COMPLETE what Backend is supposed to reply.
Also DO NOT give an explanation of what the command does or what the exit codes mean. DO NOT EVER, NOW OR IN THE FUTURE, REPLY AS BACKEND.

Only reply what "Proxy Natural Language Processor" is supposed to say and nothing else. Not now nor in the future for any reason.
```
</details>

You’ll see some good practices there, such as including lots of examples,
repetition for important behavioral aspects, constraining the replies, etc…

> :warning: Always assume that any content in a hidden prompt can be seen by the user.

### Tokens

If you thought tokens were :fire: in 2022, tokens in 2023 are on a whole
different plane of existence. The atomic unit of consumption for a language
model is not a “word”, but rather a “token”. You can kind of think of tokens
as syllables, and on average they work out to about 750 words per 1,000
tokens. They represent many concepts beyond just alphabetical characters –
such as punctuation, sentence boundaries, and the end of a document.

Here’s an example of how GPT may tokenize a sequence:

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/232417569-8d562792-64b5-423d-a7a2-db7513dd4d61.png" title="An example tokenization. You can experiment here: https://platform.openai.com/tokenizer ">
</p>

You can experiment with a tokenizer here: [https://platform.openai.com/tokenizer](https://platform.openai.com/tokenizer)

Different models will use different tokenizers with different levels of granularity. You could, in theory, just feed a model 0’s and 1’s – but then the model needs to learn the concept of characters from bits, and then the concept of words from characters, and so forth. Similarly, you could feed the model a stream of raw characters, but then the model needs to learn the concept of words, and punctuation, etc… and, in general, the models will perform worse.

To learn more, [Hugging Face has a wonderful introduction to tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) and why they need to exist.

There’s a lot of nuance around tokenization, such as vocabulary size or different languages treating sentence structure meaningfully different (e.g. words not being separated by spaces). Fortunately, language model APIs will almost always take raw text as input and tokenize it behind the scenes – *so you rarely need to think about tokens*.

**Except for one important scenario, which we discuss next: token limits.**

### Token Limits

Prompts tend to be append-only, because you want the bot to have the entire context of previous messages in the conversation. Language models, in general, are stateless and won’t remember anything about previous requests to them, so you always need to include everything that it might need to know that is specific to the current session.

A major downside of this is that the leading language model architecture, the Transformer, has a fixed input and output size – at a certain point the prompt can’t grow any larger. The total size of the prompt, sometimes referred to as the “context window”, is model dependent. For GPT-3, it is 4,096 tokens. For GPT-4, it is 8,192 tokens or 32,768 tokens depending on which variant you use.

If your context grows too large for the model, the most common tactic is the truncate the context in a sliding window fashion. If you think of a prompt as `hidden initialization prompt + messages[]`, usually the hidden prompt will remain unaltered, and the `messages[]` array will take the last N messages.

You may also see more clever tactics for prompt truncation – such as
discarding only the user messages first, so that the bot's previous answers
stay in the context for as long as possible, or asking an LLM to summarize the
conversation and then replacing all of the messages with a single message
containing that summary. There is no correct answer here and the solution will
depend on your application.

Importantly, when truncating the context, you must truncate aggressively enough to **allow room for the response as well**. OpenAI’s token limits include both the length of the input and the length of the output. If your input to GPT-3 is 4,090 tokens, it can only generate 6 tokens in response.

> 🧙‍♂️ If you’d like to count the number of tokens before sending the raw text to the model, the specific tokenizer to use will depend on which model you are using. OpenAI has a library called [tiktoken](https://github.com/openai/tiktoken/blob/main/README.md) that you can use with their models – though there is an important caveat that their internal tokenizer may vary slightly in count, and they may append other metadata, so consider this an approximation.
> 
> If you’d like an approximation without having access to a tokenizer, `input.length / 4` will give a rough, but better than you’d expect, approximation for English inputs.

### Prompt Hacking

Prompt engineering and large language models are a fairly nascent field, so new ways to hack around them are being discovered every day. The two large classes of attacks are:

1. Make the bot bypass any guidelines you have given it.
2. Make the bot output hidden context that you didn’t intend for the user to see.

There are no known mechanisms to comprehensively stop these, so it is important that you assume the bot may do or say anything when interacting with an adversarial user. Fortunately, in practice, these are mostly cosmetic concerns.

Think of prompts as a way to improve the normal user experience. **We design prompts so that normal users don’t stumble outside of our intended interactions – but always assume that a determined user will be able to bypass our prompt constraints.**

#### Jailbreaks

Typically hidden prompts will tell the bot to behave with a certain persona and focus on specific tasks or avoid certain words. It is generally safe to assume the bot will follow these guidelines for non-adversarial users, although non-adversarial users may accidentally bypass the guidelines too.

For  example, we can tell the bot:

```
You are a helpful assistant, but you are never allowed to use the word "computer".
```

If we then ask it a question about computers, it will refer to them as a “device used for computing” because it isn’t allowed to use the word “computer”.

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/232420043-ebe5bcf1-25d9-4a31-ba84-13e9e1f62de2.png" title="GPT-4 trying hard to not say the word 'computer'.">
</p>

It will absolutely refuse to say the word:

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/232420306-6fcdd6e2-b107-45d5-a1ee-4132fbb5853e.png">
</p>

But we can bypass these instructions and get the model to happily use the word if we trick it by asking it to translate the pig latin version of “computer”.

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/232420600-56083a10-b382-46a7-be18-eb9c005b8371.png">
</p>

There are [a number of defensive measures](https://learnprompting.org/docs/prompt_hacking/defensive_measures/overview) you can take here, but typically the best bet is to reiterate your most important constraints as close to the end as possible. For the OpenAI chat API, this might mean including it as a `system` message after the last `user` message. Here’s an example:

| ![image](https://user-images.githubusercontent.com/89960/232421097-adcaace3-0b21-4c1e-a5c8-46bb25faa2f7.png) | ![image](https://user-images.githubusercontent.com/89960/232421142-a47e75b4-5ff6-429d-9abd-a78dbc72466e.png) |
| --- | --- |

Despite OpenAI investing a lot into jailbreaks, there are [very clever work arounds](https://twitter.com/alexalbert__/status/1636488551817965568) being [shared every day](https://twitter.com/zswitten/status/1598088267789787136).

#### Leaks

If you missed the previous warnings in this doc, **you should always assume that any data exposed to the language model will eventually be seen by the user**.

As part of constructing prompts, you will often embed a bunch of data in hidden prompts (a.k.a. system prompts). **The bot will happily relay this information to the user**:

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/232422860-731c1de2-9e77-4957-b257-b0bbda48558c.png" title="The bot happily regurgitating the information it knows about the user.">
</p>

Even if you instruct it not to reveal the information, and it obeys those instructions, there are millions of ways to leak data in the hidden prompt.

Here we have an example where the bot should never mention my city, but a simple reframing of the question get’s it to spill the beans.

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/232423121-76568893-fa42-4ad8-b2bc-e1001327fa1e.png" title="The bot refuses to reveal personal information, but we convince it to tell me what city I’m in regardless.">
</p>

Similarly, we get the bot to tell us what word it isn’t allowed to say without ever actually saying the word:

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/232423283-1718f822-59d0-4d18-9a4d-22dd3a2672c0.png" title="Technically, the bot never said 'computer', but I was still able to get it to tell me everything I needed to know about it.">
</p>

You should think of a hidden prompt as a means to make the user experience better or more inline with the persona you’re targeting. **Never place any information in a prompt that you wouldn’t visually render for someone to read on screen**.

## Why do we need prompt engineering?

Up above, we used an analogy of prompts as the “source code” that a language model “interprets”. **Prompt engineering is the art of writing prompts to get the language model to do what we want it to do** – just like software engineering is the art of writing source code to get computers to do what we want them to do.

When writing good prompts, you have to account for the idiosyncrasies of the model(s) you’re working with. The strategies will vary with the complexity of the tasks. You’ll have to come up with mechanisms to constrain the model to achieve reliable results, incorporate dynamic data that the model can’t be trained on, account for limitations in the model’s training data, design around context limits, and many other dimensions.

There’s an old adage that computers will only do what you tell them to do. **Throw that advice out the window**. Prompt engineering inverts this wisdom. It’s like programming in natural language against a non-deterministic computer that will do anything that you haven’t guided it away from doing. 

There are two broad buckets that prompt engineering approaches fall into.

### Give a Bot a Fish

The “give a bot a fish” bucket is for scenarios when you can explicitly give the bot, in the hidden context, all of the information it needs to do whatever task is requested of it.

For example, if a user loaded up their dashboard and we wanted to show them a quick little friendly message about what task items they have outstanding, we could get the bot to summarize it as

> You have 4 receipts/memos to upload. The most recent is from Target on March 5th, and the oldest is from Blink Fitness on January 17th. Thanks for staying on top of your expenses!

by providing a list of the entire inbox and any other user context we’d like it to have.

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/233465165-e0c6b266-b347-4128-8eaa-73974e852e45.png" title="GPT-3 summarizing a task inbox.">
</p>

Similarly, if you were helping a user book a trip, you could:

- Ask the user their dates and destination.
- Behind the scenes, search for flights and hotels.
- Embed the flight and hotel search results in the hidden context.
- Also embed the company’s travel policy in the hidden context.

And then the bot will have real-time travel information + constraints that it
can use to answer questions for the user. Here’s an example of the bot
recommending options, and the user asking it to refine them:

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/233465425-9e06320c-b6d9-40ef-b5a4-c556861c1328.png" title="GPT-4 helping a user book a trip.">
</p>
<details>

  <summary>(Full prompt)</summary>

```
Brex is a platform for managing business expenses. 

The following is a travel expense policy on Brex:

- Airline highest fare class for flights under 6 hours is economy.
- Airline highest fare class for flights over 6 hours is premium economy.
- Car rentals must have an average daily rate of $75 or under.
- Lodging must have an average nightly rate of $400 or under.
- Lodging must be rated 4 stars or higher.
- Meals from restaurants, food delivery, grocery, bars & nightlife must be under $75
- All other expenses must be under $5,000.
- Reimbursements require review.

The hotel options are:
| Hotel Name | Price | Reviews |
| --- | --- | --- |
| Hilton Financial District | $109/night | 3.9 stars |
| Hotel VIA | $131/night | 4.4 stars |
| Hyatt Place San Francisco | $186/night | 4.2 stars |
| Hotel Zephyr | $119/night | 4.1 stars review |

The flight options are:
| Airline | Flight Time | Duration | Number of Stops | Class | Price |
| --- | --- | --- | --- | --- | --- |
| United | 5:30am-7:37am | 2hr 7 min | Nonstop | Economy | $248 |
| Delta | 1:20pm-3:36pm | 2hr 16 min | Nonstop | Economy | $248 |
| Alaska | 9:50pm-11:58pm | 2hr 8 min | Nonstop | Premium | $512 |

An employee is booking travel to San Francisco for February 20th to February 25th.

Recommend a hotel and flight that are in policy. Keep the recommendation concise, no longer than a sentence or two, but include pleasantries as though you are a friendly colleague helping me out:
```
 
</details>

This is the same approach that products like Microsoft Bing use to incorporate dynamic data. When you chat with Bing, it asks the bot to generate three search queries. Then they run three web searches and include the summarized results in the hidden context for the bot to use.

Summarizing this section, the trick to making a good experience is to change the context dynamically in response to whatever the user is trying to do.

> 🧙‍♂️ Giving a bot a fish is the most reliable way to ensure the bot gets a fish. You will get the most consistent and reliable results with this strategy. **Use this whenever you can.**

#### Semantic Search

If you just need the bot to know a little more about the world, [a common approach is to perform a semantic search](https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb).

A semantic search is oriented around a document embedding – which you can think of as a fixed-length array[^5] of numbers, where each number represents some aspect of the document (e.g. if it’s a science document, maybe the  843rd number is large, but if it’s an art document the 1,115th number is large – this is overly simplistic, but conveys the idea).[^6]

In addition to computing an embedding for a document, you can also compute an embedding for a user query using the same function. If the user asks “Why is the sky blue?” – you compute the embedding of that question and, in theory, this embedding will be more similar to embeddings of documents that mention the sky than embeddings that don’t talk about the sky.

To find documents related to the user query, you compute the embedding and then find the top-N documents that have the most similar embedding. Then we place these documents (or summaries of these documents) in the hidden context for the bot to reference.

Notably, sometimes user queries are so short that the embedding isn’t particularly valuable. There is a clever technique described in [a paper published in December 2022](https://arxiv.org/pdf/2212.10496.pdf) called a “Hypothetical Document Embedding” or HyDE. Using this technique, you ask the model to generate a hypothetical document in response to the user’s query, and then compute the embedding for this generated document. The model  fabricates a document out of thin air – but the approach works!

The HyDE technique uses more calls to the model, but for many use cases has notable boosts in results.

[^5]: Usually referred to as a vector.
[^6]: The vector features are learned automatically, and the specific values aren’t directly interpretable by a human without some effort.

### Teach a Bot to Fish

Sometimes you’ll want the bot to have the capability to perform actions on the user’s behalf, like adding a memo to a receipt or plotting a chart. Or perhaps we want it to retrieve data in more nuanced ways than semantic search would allow for, like retrieving the past 90 days of expenses.

In these scenarios, we need to teach the bot how to fish.

#### Command Grammars

We can give the bot a list of commands for our system to interpret, along with descriptions and examples for the commands, and then have it produce programs composed of those commands.

There are many caveats to consider when going with this approach. With complex command grammars, the bot will tend to hallucinate commands or arguments that could plausibly exist, but don’t actually. The art to getting this right is enumerating commands that have relatively high levels of abstraction, while giving the bot sufficient flexibility to compose them in novel and useful ways.

For example, giving the bot a `plot-the-last-90-days-of-expenses` command is not particularly flexible or composable in what the bot can do with it. Similarly, a `draw-pixel-at-x-y [x] [y] [rgb]` command would be far too low-level. But giving the bot a `plot-expenses` and `list-expenses` command provides some good primitives that the bot has some flexibility with.

In an example below, we use this list of commands:

| Command | Arguments | Description |
| --- | --- | --- |
| list-expenses | budget | Returns a list of expenses for a given budget |
| converse | message | A message to show to the user |
| plot-expenses | expenses[] | Plots a list of expenses |
| get-budget-by-name | budget_name | Retrieves a budget by name |
| list-budgets | | Returns a list of budgets the user has access to |
| add-memo | inbox_item_id, memo message | Adds a memo to the provided inbox item |

We provide this table to the model in Markdown format, which the language model handles incredibly well – presumably because OpenAI trains heavily on data from GitHub.

In this example below, we ask the model to output the commands in [reverse polish notation](https://en.wikipedia.org/wiki/Reverse_Polish_notation)[^7].

[^7]: The model handles the simplicity of RPN astoundingly well.

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/233505150-aef4409c-03ba-4669-95d7-6c48f3c2c3ea.png" title="A bot happily generating commands to run in response to user queries.">
</p>

> 🧠 There are some interesting subtle things going on in that example, beyond just command generation. When we ask it to add a memo to the “shake shack” expense, the model knows that the command `add-memo` takes an expense ID. But we never tell it the expense ID, so it looks up “Shake Shack” in the table of expenses we provided it, then grabs the ID from the corresponding ID column, and then uses that as an argument to `add-memo`.

Getting command grammars working reliably in complex situations can be tricky. The best levers we have here are to provide lots of descriptions, and as **many examples** of usage as we can. Large language models are [few-shot learners](https://en.wikipedia.org/wiki/Few-shot_learning_(natural_language_processing)), meaning that they can learn a new task by being provided just a few examples. In general, the more examples you provide the better off you’ll be – but that also eats into your token budget, so it’s a balance.

Here’s a more complex example, with the output specified in JSON instead of RPN. And we use Typescript to define the return types of commands.

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/233505696-fc440931-9baf-4d06-80e7-54801532d63f.png" title="A bot happily generating commands to run in response to user queries.">
</p>

<details>

  <summary>(Full prompt)</summary>
  
~~~
You are a financial assistant working at Brex, but you are also an expert programmer.

I am a customer of Brex.

You are to answer my questions by composing a series of commands.

The output types are:

```typescript
type LinkedAccount = {
    id: string,
    bank_details: {
        name: string,
        type: string,
    },
    brex_account_id: string,
    last_four: string,
    available_balance: {
        amount: number,
        as_of_date: Date,
    },
    current_balance: {
            amount: number,
        as_of_date: Date,
    },
}

type Expense = {
  id: string,
  memo: string,
  amount: number,
}

type Budget = {
  id: string,
  name: string,
  description: string,
  limit: {
    amount: number,
    currency: string,
  }
}
```

The commands you have available are:

| Command | Arguments | Description | Output Format |
| --- | --- | --- | --- |
| nth | index, values[] | Return the nth item from an array | any |
| push | value | Adds a value to the stack to be consumed by a future command | any |
| value | key, object | Returns the value associated with a key | any |
| values | key, object[] | Returns an array of values pulled from the corresponding key in array of objects | any[] |
| sum | value[] | Sums an array of numbers | number |
| plot | title, values[] | Plots the set of values in a chart with the given title | Plot |
| list-linked-accounts |  | "Lists all bank connections that are eligible to make ACH transfers to Brex cash account" | LinkedAccount[] |
| list-expenses | budget_id | Given a budget id, returns the list of expenses for it | Expense[]
| get-budget-by-name | name | Given a name, returns the budget | Budget |
| add-memo | expense_id, message | Adds a memo to an expense | bool |
| converse | message | Send the user a message | null |

Only respond with commands.

Output the commands in JSON as an abstract syntax tree.

IMPORTANT - Only respond with a program. Do not respond with any text that isn't part of a program. Do not write prose, even if instructed. Do not explain yourself.

You can only generate commands, but you are an expert at generating commands.
~~~

</details>

This version is a bit easier to parse and interpret if your language of choice has a `JSON.parse` function.

> 🧙‍♂️ There is no industry established best format for defining a DSL for the model to generate programs. So consider this an area of active research. You will bump into limits. And as we overcome these limits, we may discover more optimal ways of defining commands.

#### ReAct

In March of 2023, Princeton and Google released a paper “[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/pdf/2210.03629.pdf)”, where they introduce a variant of command grammars that allows for fully autonomous interactive execution of actions and retrieval of data.

The model is instructed to return a `thought` and an `action` that it would like to perform. Another agent (e.g. our client) then performs the `action` and returns it to the model as an `observation`. The model will then loop to return more thoughts and actions until it returns an `answer`.

This is an incredibly powerful technique, effectively allowing the bot to be its own research assistant and possibly take actions on behalf of the user. Combined with a powerful command grammar, the bot should rapidly be able to answer a massive set of user requests.

In this example, we give the model a small set of commands related to getting employee data and searching wikipedia:

| Command | Arguments | Description |
| --- | --- | --- |
| find_employee | name | Retrieves an employee by name |
| get_employee | id | Retrieves an employee by ID |
| get_location | id | Retrieves a location by ID |
| get_reports | employee_id | Retrieves a list of employee ids that report to the employee associated with employee_id. |
| wikipedia | article | Retrieves a wikipedia article on a topic. |

We then ask the bot a simple question, “Is my manager famous?”.

We see that the bot:

1. First looks up our employee profile.
2. From our profile, gets our manager’s id and looks up their profile.
3. Extracts our manager’s name and searches for them on Wikipedia.
    - I chose a fictional character for the manager in this scenario.
4. The bot reads the wikipedia article and concludes that can’t be my manager since it is a fictional character.
5. The bot then modifies its search to include (real person).
6. Seeing that there are no results, the bot concludes that my manager is not famous.

| ![image](https://user-images.githubusercontent.com/89960/233506839-5c8b2d77-1d78-464d-bc33-a725e12f2624.png) | ![image](https://user-images.githubusercontent.com/89960/233506870-05fc415d-efa2-48b7-aad9-b5035e535e6d.png) |
| --- | --- |

<details>
<summary>(Full prompt)</summary>

~~~
You are a helpful assistant. You run in a loop, seeking additional information to answer a user's question until you are able to answer the question.

Today is June 1, 2025. My name is Fabian Seacaster. My employee ID is 82442.

The commands to seek information are:

| Command | Arguments | Description |
| --- | --- | --- |
| find_employee | name | Retrieves an employee by name |
| get_employee | id | Retrieves an employee by ID |
| get_location | id | Retrieves a location by ID |
| get_reports | employee_id | Retrieves a list of employee ids that report to the employee associated with `employee_id`. |
| wikipedia | article | Retrieves a wikipedia article on a topic. |

Your response will be in JSON and will include a "Thought" + "Action" to retrieve data that you need in order to answer the question, or it will include the "Answer". When data has been retrieved, it will be included as an "Observation".

You will continue generating thoughts and actions until you get to an answer, or conclude that you can't.

Example 1:
```
User: What is the population of Philadelphia?

Assistant: {
  "thought": "Wikipedia likely has this information. I'm looking it up...",
  "action": {"command": "wikipedia", "article": "Philadelphia"}
}

Assistant: {
  "observation": "Philadelphia, often called Philly, is the largest city in the Commonwealth of Pennsylvania and the second-largest city in both the Northeast megalopolis and Mid-Atlantic regions after New York City. It is one of the most historically significant cities in the United States and served as the nation's capital city until 1800. Philadelphia is the nation's sixth-largest city with a population of 1,603,797 as of the 2020 census. Since 1854, the city has been coextensive with Philadelphia County, the most populous county in Pennsylvania and the urban core of the Delaware Valley, the nation's seventh-largest and one of the world's largest metropolitan regions with 6.245 million residents in 2020. Philadelphia is known for its extensive contributions to American history and for its role in the life sciences, business and industry, art, literature, and music."
}

Assistant: {
  "answer": "The population of Philadelphia, as of the 2020 census, is 1,603,797 people."
}
```

Example 2:
```
User: What is my manager's city?

Assistant: {
  "thought": "Let me look up who your manager is. First, I need to look up your profile.",
  "action": {"command": "get_employee", "id": 92352}
}

Assistant: {
  "observation": {
    "id": 78334,
    "name": "Ms. Manager",
    "location_id": 8832
  }
}

Assistant: {
  "thought": "Your manager is Ms. Manager. I'm looking up their location.",
  "action": {"command": "get_location", "id": 8832}
}

Assistant: {
  "observation": {
    "id": 8832,
    "name": "Philadelphia"
  }
}

Assistant: {
  "answer": "Your manager lives in Philadelphia."
}
```
~~~
</details>

#### GPT-4 vs GPT-3.5

In most of the examples in this doc, the difference between GPT-3.5 and GPT-4 is negligible, but for “teaching a bot to fish” scenarios the difference between the models is notable.

None of the above examples of command grammars, for example, work without meaningful modifications for GPT-3.5. At a minimum, you have to provide a number of examples (at least one usage example per command) before you get any reasonable results. And, for complex sets of commands, it may hallucinate new commands or create fictional arguments.

With a sufficiently thorough hidden prompt, you should be able to overcome these limitations. GPT-4 is capable of far more consistent and complex logic with far simpler prompts (and can get by with zero or  small numbers of examples – though it is always beneficial to include as many as possible).

## Strategies

This section contains examples and strategies for specific needs or problems. For successful prompt engineering, you will need to combine some subset of all of the strategies enumerated in this document. Don’t be afraid to mix and match things – or invent your own approaches.

### Embedding Data

In hidden contexts, you’ll frequently want to embed all sorts of data. The specific strategy will vary depending on the type and quantity of data you are embedding.

#### Simple Lists

For one-off objects, enumerating fields + values in a normal bulleted list works pretty well:

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/233507156-0bdbc0af-d977-44e0-a8d5-b30538c5bbd9.png" title="GPT-4 extracting Steve’s occupation from a list attributes.">
</p>

It will also work for larger sets of things, but there are other formats for lists of data that GPT handles more reliably. Regardless, here’s an example:

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/233507223-9cda591e-62f3-4339-b227-a07c37b90724.png" title="GPT-4 answering questions about a set of expenses.">
</p>

#### Markdown Tables

Markdown tables are great for scenarios where you have many items of the same type to enumerate.

Fortunately, OpenAI’s models are exceptionally good at working with Markdown tables (presumably from the tons of GitHub data they’ve trained on).

We can reframe the above using Markdown tables instead:

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/233507313-7ccd825c-71b9-46d3-80c9-30bf97a8e090.png" title="GPT-4 answering questions about a set of expenses from a Markdown table.">
</p>

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/233507395-b8ecb641-726c-4e57-b85e-13f6b7717f22.png" title="GPT-4 answering questions about a set of expenses from a Markdown table.">
</p>

> 🧠 Note that in this last example, the items in the table have an explicit date, February 2nd. In our question, we asked about “today”. And earlier in the prompt we mentioned that today was Feb 2. The model correctly handled the transitive inference – converting “today” to “February 2nd” and then looking up “February 2nd” in the table.

#### JSON

Markdown tables work really well for many use cases and should be preferred due to their density and ability for the model to handle them reliably, but you may run into scenarios where you have many columns and the model struggles with it or every item has some custom attributes and it doesn’t make sense to have dozens of columns of empty data.

In these scenarios, JSON is another format that the model handles really well. The close proximity of `keys` to their `values` makes it easy for the model to keep the mapping straight.

Here is the same example from the Markdown table, but with JSON instead:

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/233507559-26e6615d-4896-4a2c-b6ff-44cbd7d349dc.png" title="GPT-4 answering questions about a set of expenses from a JSON blob.">
</p>

#### Freeform Text

Occasionally you’ll want to include freeform text in a prompt that you would like to delineate from the rest of the prompt – such as embedding a document for the bot to reference. In these scenarios, surrounding the document with triple backticks, ```, works well[^8].

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/233507684-93222728-e216-47b4-8554-04acf9ec6201.png" title="GPT-4 answering questions about a set of expenses from a JSON blob.">
</p>

[^8]: A good rule of thumb for anything you’re doing in prompts is to lean heavily on things the model would have learned from GitHub.

#### Nested Data

Not all data is flat and linear. Sometimes you’ll need to embed data that is nested or has relations to other data. In these scenarios, lean on `JSON`:

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/233507758-7baffcaa-647b-4869-9cfb-a7cf8849c453.png" title="GPT-4 handles nested JSON very reliably.">
</p>

<details>
<summary>(Full prompt)</summary>

~~~
You are a helpful assistant. You answer questions about users. Here is what you know about them:

{
  "users": [
    {
      "id": 1,
      "name": "John Doe",
      "contact": {
        "address": {
          "street": "123 Main St",
          "city": "Anytown",
          "state": "CA",
          "zip": "12345"
        },
        "phone": "555-555-1234",
        "email": "johndoe@example.com"
      }
    },
    {
      "id": 2,
      "name": "Jane Smith",
      "contact": {
        "address": {
          "street": "456 Elm St",
          "city": "Sometown",
          "state": "TX",
          "zip": "54321"
        },
        "phone": "555-555-5678",
        "email": "janesmith@example.com"
      }
    },
    {
      "id": 3,
      "name": "Alice Johnson",
      "contact": {
        "address": {
          "street": "789 Oak St",
          "city": "Othertown",
          "state": "NY",
          "zip": "67890"
        },
        "phone": "555-555-2468",
        "email": "alicejohnson@example.com"
      }
    },
    {
      "id": 4,
      "name": "Bob Williams",
      "contact": {
        "address": {
          "street": "135 Maple St",
          "city": "Thistown",
          "state": "FL",
          "zip": "98765"
        },
        "phone": "555-555-8642",
        "email": "bobwilliams@example.com"
      }
    },
    {
      "id": 5,
      "name": "Charlie Brown",
      "contact": {
        "address": {
          "street": "246 Pine St",
          "city": "Thatstown",
          "state": "WA",
          "zip": "86420"
        },
        "phone": "555-555-7531",
        "email": "charliebrown@example.com"
      }
    },
    {
      "id": 6,
      "name": "Diane Davis",
      "contact": {
        "address": {
          "street": "369 Willow St",
          "city": "Sumtown",
          "state": "CO",
          "zip": "15980"
        },
        "phone": "555-555-9512",
        "email": "dianedavis@example.com"
      }
    },
    {
      "id": 7,
      "name": "Edward Martinez",
      "contact": {
        "address": {
          "street": "482 Aspen St",
          "city": "Newtown",
          "state": "MI",
          "zip": "35742"
        },
        "phone": "555-555-6813",
        "email": "edwardmartinez@example.com"
      }
    },
    {
      "id": 8,
      "name": "Fiona Taylor",
      "contact": {
        "address": {
          "street": "531 Birch St",
          "city": "Oldtown",
          "state": "OH",
          "zip": "85249"
        },
        "phone": "555-555-4268",
        "email": "fionataylor@example.com"
      }
    },
    {
      "id": 9,
      "name": "George Thompson",
      "contact": {
        "address": {
          "street": "678 Cedar St",
          "city": "Nexttown",
          "state": "GA",
          "zip": "74125"
        },
        "phone": "555-555-3142",
        "email": "georgethompson@example.com"
      }
    },
    {
      "id": 10,
      "name": "Helen White",
      "contact": {
        "address": {
          "street": "852 Spruce St",
          "city": "Lasttown",
          "state": "VA",
          "zip": "96321"
        },
        "phone": "555-555-7890",
        "email": "helenwhite@example.com"
      }
    }
  ]
}
~~~
</details>

If using nested `JSON` winds up being too verbose for your token budget, fallback to `relational tables` defined with `Markdown`:

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/233507968-a378587b-e468-4882-a1e8-678d9f3933d3.png" title="GPT-4 handles relational tables pretty reliably too.">
</p>

<details>
<summary>(Full prompt)</summary>

~~~
You are a helpful assistant. You answer questions about users. Here is what you know about them:

Table 1: users
| id (PK) | name          |
|---------|---------------|
| 1       | John Doe      |
| 2       | Jane Smith    |
| 3       | Alice Johnson |
| 4       | Bob Williams  |
| 5       | Charlie Brown |
| 6       | Diane Davis   |
| 7       | Edward Martinez |
| 8       | Fiona Taylor  |
| 9       | George Thompson |
| 10      | Helen White   |

Table 2: addresses
| id (PK) | user_id (FK) | street      | city       | state | zip   |
|---------|--------------|-------------|------------|-------|-------|
| 1       | 1            | 123 Main St | Anytown    | CA    | 12345 |
| 2       | 2            | 456 Elm St  | Sometown   | TX    | 54321 |
| 3       | 3            | 789 Oak St  | Othertown  | NY    | 67890 |
| 4       | 4            | 135 Maple St | Thistown  | FL    | 98765 |
| 5       | 5            | 246 Pine St | Thatstown  | WA    | 86420 |
| 6       | 6            | 369 Willow St | Sumtown  | CO    | 15980 |
| 7       | 7            | 482 Aspen St | Newtown   | MI    | 35742 |
| 8       | 8            | 531 Birch St | Oldtown   | OH    | 85249 |
| 9       | 9            | 678 Cedar St | Nexttown  | GA    | 74125 |
| 10      | 10           | 852 Spruce St | Lasttown | VA    | 96321 |

Table 3: phone_numbers
| id (PK) | user_id (FK) | phone       |
|---------|--------------|-------------|
| 1       | 1            | 555-555-1234 |
| 2       | 2            | 555-555-5678 |
| 3       | 3            | 555-555-2468 |
| 4       | 4            | 555-555-8642 |
| 5       | 5            | 555-555-7531 |
| 6       | 6            | 555-555-9512 |
| 7       | 7            | 555-555-6813 |
| 8       | 8            | 555-555-4268 |
| 9       | 9            | 555-555-3142 |
| 10      | 10           | 555-555-7890 |

Table 4: emails
| id (PK) | user_id (FK) | email                 |
|---------|--------------|-----------------------|
| 1       | 1            | johndoe@example.com   |
| 2       | 2            | janesmith@example.com |
| 3       | 3            | alicejohnson@example.com |
| 4       | 4            | bobwilliams@example.com |
| 5       | 5            | charliebrown@example.com |
| 6       | 6            | dianedavis@example.com |
| 7       | 7            | edwardmartinez@example.com |
| 8       | 8            | fionataylor@example.com |
| 9       | 9            | georgethompson@example.com |
| 10      | 10           | helenwhite@example.com |

Table 5: cities
| id (PK) | name         | state | population | median_income |
|---------|--------------|-------|------------|---------------|
| 1       | Anytown     | CA    | 50,000     | $70,000      |
| 2       | Sometown    | TX    | 100,000    | $60,000      |
| 3       | Othertown   | NY    | 25,000     | $80,000      |
| 4       | Thistown    | FL    | 75,000     | $65,000      |
| 5       | Thatstown   | WA    | 40,000     | $75,000      |
| 6       | Sumtown     | CO    | 20,000     | $85,000      |
| 7       | Newtown     | MI    | 60,000     | $55,000      |
| 8       | Oldtown     | OH    | 30,000     | $70,000      |
| 9       | Nexttown    | GA    | 15,000     | $90,000      |
| 10      | Lasttown    | VA    | 10,000     | $100,000     |
~~~

</details>

> 🧠 The model works well with data in [3rd normal form](https://en.wikipedia.org/wiki/Third_normal_form), but may struggle with too many joins. In experiments, it seems to do okay with at least three levels of nested joins. In the example above the model successfully joins from `users` to `addresses` to `cities` to infer the likely income for George – $90,000.

### Citations

Frequently, a natural language response isn’t sufficient on its own and you’ll want the model’s output to cite where it is getting data from. 

One useful thing to note here is that anything you might want to cite should have a unique ID. The simplest approach is to just ask the model to link to anything it references:


<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/233509069-1dcbffa2-8357-49b5-be43-9791f93bd0f8.png" title="GPT-4 will reliably link to data if you ask it to.">
</p>

### Programmatic Consumption

By default, language models output natural language text, but frequently we need to interact with this result in a programmatic way that goes beyond simply printing it out on screen. You can achieve this by  asking the model to output the results in your favorite serialization format (JSON and YAML seem to work best).

Make sure you give the model an example of the output format you’d like. Building on our previous travel example above, we can augment our prompt to tell it:

~~~
Produce your output as JSON. The format should be:
```
{
    message: "The message to show the user",
    hotelId: 432,
    flightId: 831
}
```

Do not include the IDs in your message.
~~~

And now we’ll get interactions like this:

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/233509174-be0c3bc5-08e3-4d1a-8841-52c401def770.png" title="GPT-4 providing travel recommendations in an easy to work with format.">
</p>

You could imagine the UI for this rendering the message as normal text, but then also adding discrete buttons for booking the flight + hotel, or auto-filling a form for the user.

As another example, let’s build on the [citations](#citations) example – but move beyond Markdown links. We can ask it to produce JSON with a normal message along with a list of items used in the creation of that message. In this scenario you won’t know exactly where in the message the citations were leveraged, but you’ll know that they were used somewhere.

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/233509280-59d9ff46-0e95-488a-b314-a7d2b7c9bfa3.png" title="Asking the model to provide a list of citations is a reliable way to programmatically know what data the model leaned on in its response.">
</p>

> 🧠 Interestingly, in the model’s response to “How much did I spend at Target?” it provides a single value, $188.16, but **importantly** in the `citations` array it lists the individual expenses that it used to compute that value.

### Chain of Thought

Sometimes you will bang your head on a prompt trying to get the model to output reliable results, but, no matter what you do, it just won’t work. This will frequently happen when the bot’s final output requires intermediate thinking, but you ask the bot only for the output and nothing else.

The answer may surprise you: ask the bot to show its work. In October 2022, Google released a paper “[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/pdf/2201.11903.pdf)” where they showed that if, in your hidden prompt, you give the bot examples of answering questions by showing your work, then when you ask the bot to answer something it will show its work and produce more reliable answers.

Just a few weeks after that paper was published, at the end of October 2022, the University of Tokyo and Google released the paper “[Large Language Models are Zero-Shot Reasoners](https://openreview.net/pdf?id=e2TBb5y0yFf)”, where they show that you don’t even need to provide examples – **you simply have to ask the bot to think step-by-step**.

#### Averaging

Here is an example where we ask the bot to compute the average expense, excluding Target. The actual answer is $136.77 and the bot almost gets it correct with $136.43.

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/233509534-2b32c8dd-a1ee-42ea-82fb-4f84cfe7e9ba.png" title="The model **almost** gets the average correct, but is a few cents off.">
</p>

If we simply add “Let’s think step-by-step”, the model gets the correct answer:

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/233509608-6e53995b-668b-47f6-9b5e-67afad89f8bc.png" title="When we ask the model to show its work, it gets the correct answer.">
</p>

#### Interpreting Code

Let’s revisit the Python example from earlier and apply chain-of-thought prompting to our question. As a reminder, when we asked the bot to evaluate the Python code it gets it slightly wrong. The correct answer is `Hello, Brex!!Brex!!Brex!!!` but the bot gets confused about the number of !'s to include. In below’s example, it outputs `Hello, Brex!!!Brex!!!Brex!!!`:

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/233509724-8f3302f8-59eb-4d3b-8939-53d7f63b0299.png" title="The bot almost interprets the Python code correctly, but is a little off.">
</p>

If we ask the bot to show its work, then it gets the correct answer:

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/233509790-2a0f2189-d864-4d27-aacb-cfc936fad907.png" title="The bot correctly interprets the Python code if you ask it to show its work.">
</p>

#### Delimiters

In many scenarios, you may not want to show the end user all of the bot’s thinking and instead just want to show the final answer. You can ask the bot to delineate the final answer from its thinking. There are many ways to do this, but let’s use JSON to make it easy to parse:

<p align="center">
  <img width="550" src="https://user-images.githubusercontent.com/89960/233509865-4f3e7265-6645-4d43-8644-ecac5c0ca4a7.png" title="The bot showing its work while also delimiting the final answer for easy extraction.">
</p>

Using Chain-of-Thought prompting will consume more tokens, resulting in increased price and latency, but the results are noticeably more reliable for many scenarios. It’s a valuable tool to use when you need the bot to do something complex and as reliably as possible.

### Fine Tuning

Sometimes no matter what tricks you throw at the model, it just won’t do what you want it to do. In these scenarios you can **sometimes** fallback to fine-tuning. This should, in general, be a last resort.

[Fine-tuning](https://platform.openai.com/docs/guides/fine-tuning) is the process of taking an already trained model and then giving it thousands (or more) of example `input:output` pairs

It does not eliminate the need for hidden prompts, because you still need to embed dynamic data, but it may make the prompts smaller and more reliable.

#### Downsides

There are many downsides to fine-tuning. If it is at all possible, take advantage of the nature of language models being [zero-shot, one-shot, and few-shot learners](https://en.wikipedia.org/wiki/Few-shot_learning_(natural_language_processing)) by teaching them to do something in their prompt rather than fine-tuning.

Some of the downsides include:

- **Not possible**: [GPT-3.5/GPT-4 isn’t fine tunable](https://platform.openai.com/docs/guides/chat/is-fine-tuning-available-for-gpt-3-5-turbo), which is the primary model / API we’ll be using, so we simply can’t lean in fine-tuning.
- **Overhead**: Fine-tuning requires manually creating tons of data.
- **Velocity**: The iteration loop becomes much slower – every time you want to add a new capability, instead of adding a few lines to a prompt, you need to create a bunch of fake data and then run the finetune process and then use the newly fine-tuned model.
- **Cost**: It is up to 60x more expensive to use a fine-tuned GPT-3 model vs the stock `gpt-3.5-turbo` model. And it is 2x more expensive to use a fine-tuned GPT-3 model vs the stock GPT-4 model.

> ⛔️ If you fine-tune a model, **never use real customer data**. Always use synthetic data. The model may memorize portions of the data you provide and may regurgitate private data to other users that shouldn’t be seeing it.
>
> If you never fine-tune a model, we don’t have to worry about accidentally leaking data into the model.

## Additional Resources
- :star2: [OpenAI Cookbook](https://github.com/openai/openai-cookbook) :star2:
- :technologist: [Prompt Hacking](https://learnprompting.org/docs/category/-prompt-hacking) :technologist: 
- :books: [Dair.ai Prompt Engineering Guide](https://github.com/dair-ai/Prompt-Engineering-Guide) :books: 
