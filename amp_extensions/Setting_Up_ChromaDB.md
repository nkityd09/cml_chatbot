# Setting UP Independent ChromaDB

I have tested the deployment of Chroma on a long-running server, and connect to it remotely via the CML Application. Chroma provides a CloudFormation stack which can be used to deploy an EC2 instance with Chroma running on it via Docker. The application is accessible via port 8000 and the Public IP address of the instance.

The official documentation and steps can be found [here](https://docs.trychroma.com/deployment).

## Steps to deploy standalone ChromaDB

1. Copy the below S3 URL which stores the Chroma Cloudformation Stack
```
https://s3.amazonaws.com/public.trychroma.com/cloudformation/latest/chroma.cf.json
```

2. Navigate to AWS Console and the CloudFormation Service
3. Click “Create Stack” “with new resources” and provide the above URL under “Amazon S3 URL”
4. Give the stack a name, optionally change the Instance Type, if needed. Provide a Key pair name which can be used to access the EC2 instance
5. Add Tags if required and then Click “Next”
6. Review all settings and Click “Submit”
7. Once the stack creation  completes successfully, navigate to the Outputs Tab to see the Public IP of the instance

![Chroma CloudFormation](../images/chroma_cfn.png)

Make a note of the Public IP from the Outputs tab of Cloudformation. This will be needed when launching the CML AMP.


