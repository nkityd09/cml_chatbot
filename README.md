# CML PDF Document Chatbot
Git Repo for CML Document Chatbot

## Features Under Development

1. Independent VectorDB 
   - [ChromaDB](https://docs.trychroma.com/)  **Tested Successfully** -> [Documentation](amp_extensions/Setting_Up_ChromaDB.md)
   - [Milvus](https://milvus.io/) (**In Progress**)
2. [Llama-2 7B and 13B Chat LLMs](https://huggingface.co/meta-llama)  **Tested Successfully** -> [Documentation](amp_extensions/Using_Llama-2.md)
3. Conversational Memory (**In Progress**)

## Resource Requirements

The AMP Application has been configured to use the following 
- 4 CPU
- 32 GB RAM
- 2 GPU

## Steps to Setup the CML AMP Application

1. Navigate to CML Workspace -> Site Administration -> AMPs Tab

2. Under AMP Catalog Sources section, We will "Add a new source By" selecting "Catalog File URL" 

3. Provide the following URL and click "Add Source"

```
https://raw.githubusercontent.com/nkityd09/cml_chatbot/main/catalog.yaml
```

4. Once added, We will be able to see the LLM PDF Document Chatbot in the AMP section and deploy it from there.

5. Click on the AMP and "Configure Project", disable Spark as it is not required.

6. Once the AMP steps are completed, We can access the Gradio UI via the Applications page.


## Steps to Use the Gradio App

1. Navigate to the "Upload File" Tab and use the "Click to Upload Button" to upload a file

![Uploading Files](images/File_Upload.png)

2. Once the files have been uploaded, use the "Embed Document" button to store the document into VectorDB

**Note Embedding documents is lenghty process and can take some time to complete.**

![Embedding Files](images/File_Embed.png)

3. Once Embedding has completed, switch to the FileGPT tab and enter your questions via the textbox and Submit button below.

![Asking Questions](images/Response.png)
