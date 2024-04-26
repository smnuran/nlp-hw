
This homework is about writing questions that are interesting and that
a reasonable human could answer but that computers struggle to answer.

Tools to Write Your Question
=======

You are encouraged (but not required) to use this tool:
https://qantanew.web.app/play
This app has a task of writing an ADVERSARIAL question about the topic you chose. 
This question should be writen so that it is answerable with the page title. All the machine guesses surrounding the page content
are updated as you write the questions, and this way you are informed if you are doing a good job of fooling the computer or not!
(It has a built in tutorial, but we're looking to improve it, so please give us feedback on it!)

Here are some techniques to write questions systems cannot answer:
https://www.youtube.com/watch?v=6oZCIOBiSaI

You can see a previous competition we ran here:
https://sites.google.com/view/qanta/past-events/dec-15-2018?authuser=0

You are however free to download/use any QA systems you'd like or to use them on the web.  Some ideas include:
* https://you.com/
* https://www.bing.com/new
* https://chat.openai.com/

Please share any good ones you find with the rest of the class.

Format of the Question
========

As discussed in class, the questions will be pyramidal, since this is the format we'll use for the course project.  
Some resources for writing questions:
https://www.naqt.com/resources/question-writing.html

How to describe your question
=========
A question requires about a page of explanation with the following subsections:
* _Question_: The question you asked, why you thought the question is difficult for computers while not as hard for the humans, why you chose this specific topic to ask a question about. If it is pyramidial, why choose to do so? 
* _Guess_: What answer systems provided (make sure to provide details on what systems you tried against). The details may include why you chose this answer system.   
* _Answer_: What the correct answer should be.  Provide sources with citations.  Wikipedia should not be used as a primary source.  Better sources are peer-reviewed articles (JSTOR is your friend and is available from UMD IPs without a fee), newspapers, books (Google books is good), etc.
* _Guess Explanation_: Why did the AI provide the answer that it did? Connect it with why you chose this particular answer system and if the guesss it gave matches with your intent of making the question difficult. For the
homework, this can be in general terms (but do make use of concepts like
semantic/syntactic ambiguity that we've covered in class), but for the project
this should be backed up with relevant citations from the NLP literature. 
* _Interesting/Notability_: Why would someone find this question interesting
or why would someone want to know the answer to this question?  Use citations
as relevant/necessary.  Possible explanations are: surprisingness (Brian May
from Queen has a PhD), novelty (John Tyler is the earliest former
President of the US with a living grandchild), importance (the Witting
reaction is taught in most organic chemistry classes), or connective (Samuel
Beckett drove Andre the Giant to School).

There's an example of this in this directory.  The citations do not count against the page limit, and it's okay to go slightly beyond a page.

Use of ChatGPT
========

Just so there are no mixed signals, you're allowed (and encouraged) to use
ChatGPT and other AI tools to check your work.  However, don't try to use them
to generate questions or explanations (it probably won't work).  In general,
adhere to the policy articulated here:
https://2023.aclweb.org/blog/ACL-2023-policy/


Groups
=========

You can work in groups for this assignment, but it doesn't decrease the
workload (e.g., if you're N people, you must submit N questions).  Everyone
will get the same score for their submission, and if late days are used it
will be applied to all members of the group.

What to turn in
==========

Turn in both a PDF of the writeup of your question and a json including the raw text of the question and answer to Gradescope. Both examples of both writeup and the json file are included in this directory.

Grading
==========

You will lose points if your question is:
* Vague
* Has incorrect information
* Has incorrect grammar
* Makes it seem like it's asking for two different things
* Is a near copy of an existing question

You will lose points if your answer is:
* Too obscure for anyone to know

Good questions will get full points if:
* The question is difficult with an accessible answer
* Clue-dense
* Reward knowledge
* Have specific facts
* Stump computers
* Are answerable by any reasonably-informed humans

Writeups will get full points if they:
* Cite all facts in the question with good (i.e., not Wikipedia) references
* Clearly explain why they structured the information and the pyramidality
  

FAQ
===========

*Q:* Can I submit multiple questions?

*A:* No, submit your best question.

*Q:* What humans should be able to answer the question?  Is it okay if it's hard?

*A:* At the minimum, a skilled trivia player should be able to get the question right.  However, it's even better if more people can get the question right.  The easier the question is for an average human (let's assume American undergrad for the purposes of this exercise), the more impressive it is that a computer cannot answer it.  

*Q:* Do I have to use a Wikipedia page title as the answer?

*A:* Normally yes.  It helps to standardize and will also let us link it to resources.  However, some of the answers are not going to be in that set.  E.g., if you want to have an answer like "missing a leg" or "ways Sean Bean has died in films" or "because they're all dead", that's not going to match a page.  Please try to match if you can, but otherwise, an arbitrary string is fine.

*Q:* Can the answer be a year or a number?

*A:* Yes, but these questions are often difficult to write so that they're uniquely identifying.  For events that are ancient history, there's often debate about when something happened.  And there are different calendar systems.  You you'd need to specify something like "Based on the Greogorian calendar".  You also need to be specific about if you want a year, a day, a month, etc.  There's also sometimes confusion about when something happened: battles can last days, elections are voted on in November but someone isn't sworn in until the following January.  So all of this is to say, it's okay to ask this, but it sometimes requires more care.

Also, humans don't memorize a lot of dates.  Some things are tightly tied to dates (coronation of Charlemagne, Pearl Harbor, September 11), but most things are not.  So if we're looking for things that humans can answer but computers cannot, these sorts of things may be more difficult.

*Q:* What computers should ~not~ be able to answer the questions?

*A:* It's okay if some computers can't answer the questions.  But clearly it would be better if all computers cannot answer the question.  Even better is if they all fail in different ways.  We're not going to focus on particular systems and say that, for instance, it must absolutely stump ChatGPT.

*Q*: Is it sufficient to just fool the interface?

*A*: The (authoring interface)[https://qantanew.web.app/login] is just one QA system, and
 writing a question that stumps it is good, but there are other
 systems out there.  We hope that an advantage of using it will be
 that it will provide you better feedback both on the evidence that
 it's finding as well as providing better clues about how pyramidal
 the question is.  But you should use a variety of systems to check
 your questions and answers.

*Q*: Why does the interface want access to by Google Docs?

*A*: It wants access to Google Docs so you can edit your questions
 there and load them in this interface if you prefer.  
