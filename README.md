# cml_chatbot
Git Repo for CML Chatbot

## Features Under Development

1. Independent VectorDB
2. Conversational Memory

**Note:** This AMP has been tested with Llama-2. The steps to setup Llama-2 will be added to the README shortly.

## Resource Requirements

The AMP Application has been configured to use the following 
- 4 CPU
- 32 GB RAM
- 2 GPU

## Steps to Setup the CML AMP Application

Step 1.
Navigate to CML Workspace -> Site Administration -> AMPs Tab

Step 2

Under AMP Catalog Sources section, We will "Add a new source By" selecting "Catalog File URL" 

Step 3

Provide the following URL and click "Add Source"

```
https://raw.githubusercontent.com/nkityd09/cml_chatbot/main/catalog.yaml
```

Step 4

Once added, We will be able to see the LLM PDF Document Chatbot in the AMP section and deploy it from there.

Step 5 

Click on the AMP and "Configure Project", disable Spark as it is not required.

Step 6

Once the AMP steps are completed, We can access the Gradio UI via the Applications page.


## Steps to Use the Gradio App

Step 1

Navigate to the "Upload File" Tab and use the "Click to Upload Button" to upload a file

Step 2

Once the files have been uploaded, use the "Embed Document" button to store the document into VectorDB
**Note** Embedding documents is lenghty process and can take some time to complete.

Step 3

Once Embedding has completed, switch to the FileGPT tab and enter your questions via the textbox and Submit button below.
