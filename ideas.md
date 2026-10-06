
# MBPP (Mostly Basic Python Problems)


## High level breakdown

Given a set of problems from MBPP:
- Generate a candiate solution (using an AI agent)
- Pass the proposed solution into an ML black box
- Get trustworthiniess score out of it


## Project idea 1:

Using only the results of the patch out come such as:
    - Did it run?
    - Did it return the valid resutls?
    - Other runtime results

Pass it into a non-deep learning ML approach to retrun if the patch is good or bad.

Concerns: That is not any different that the inital POC from the SWE-Bench.
Code was generated was run against the harness then trained on the simple regression metrics
to about an accuracy of 74% (verified) or 69% (full)...
Doing idea 1 is not meaningfully different than the POC and does not solve the underlying issue brought from there that "there needs to be another metric inserted to improve accuracy"

## Project Idea 2:

Use the data as the datatype it is:
    - Input problem statement 
    - Input the proposed patch
    - Insert simple run time metrics (from idea 1) 
  

This can be done with standard transformer encoder architecture by:
    - Tokenizing problem statement 
    - Tokenizing proposed patch
    - Compressing runtime output metrics into embedding space then into a token appended
  
What the model does:
    - Ingest tokens
    - applies the attention mechanism
    - leverages a projection head
    - return a trustworithniess score (aka probability)
  


## Things needed regardless

**Data**.

This project over all requires a lot of data that is not readily available. 
We can and should use AI agents/LLMs whatever to generate patches with "control".

What control means is that we tell it to generate good or bad patches with specific 
failure modes to include. (i.e. misspelling a variable so it crashes or other things that would be
realistic in a real world environment)

How much data we need is an unknown quantity. More is better, but the constraining factor for this project is the problem statements. 

But we need to pick a number and stick with it. 
If we can realistically take the 1000 MBPP and generate 10 cases for each problem then that 
is the constrained dataset. We just need to idenitfy what data was collect and why (ID the bias, etc).

**Datasets**

The SWE-Bench is good for verbose, but verifying is not ideal for the timeline constrain
MBPP is good for verifying, but lacks verboseness. 

There are trade offs and there is not right or wrong answer, but we need to pick and stick. 
 

**Results**

At the end of the day because a dataset for this does not exist; expectations need to be managed.

If we could ingest and verify against nearly every single public github repo with even roughly labeled data would improve perofrmance sigficantly. Using SWE-Bench or MBPP is only going to do so much. At the end of the day we are not trying to change the world, we are attempting to demonstrate undestanding of the Machine Learning Engieering principles, processes, and pipelines. 


## Recomendation

Focus:**MBPP**
Justification: Small and easy to genearte a small scale dataset that can be leveraged with available hardware and within class time constrains

Model Idea: **Both**
Justification: Cross verifying why certain models work better than others is a key aspect of MLE and research in generale. Additionaly code infrustruture for project idea 1 exists and can be refurbished. 

Dataset size: **No more than 50K samples**
Justification: While MBPP is short so the dataset size is not a concern. There are so many solutions correct or incorrect that a "basic" problem can have. So 50K gets about 50 examples per problem. 


Actions to be take:
- Determine a set of constrains to feed into the generating LLM to either have it produce a:
  - Valid patch
  - Invalid patch with known issues
- Determine which set of models we can run to generate test patches 
  - Constrain: If CW personal system is used only non-FA models can be used to generate patches.

