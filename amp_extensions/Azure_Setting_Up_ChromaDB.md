# Installing Docker and Chroma on RHEL 8.7 in Azure

## 1. Install required packages for Docker:

```
sudo yum install -y yum-utils
```

## 2. Add the Docker CE repository:
```
sudo yum-config-manager --add-repo https://download.docker.com/linux/rhel/docker-ce.repo
```

## 3. Navigate to the yum repositories directory:
```
cd /etc/yum.repos.d
```

## 4. Edit the Docker CE repository file:
```
sudo vi docker-ce.repo 
```
Inside the vi editor:

a. Locate the section [docker-ce-stable].

b. Modify the section to match the following content:

```
name=Docker CE Stable - $basearch
baseurl=https://download.docker.com/linux/centos/$releasever/$basearch/stable
enabled=1
gpgcheck=1
gpgkey=https://download.docker.com/linux/centos/gpg
```

Save and exit the editor (:wq in vi).


## 5. Install Docker and associated components:
```
sudo yum install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

## 6. Install Git (used to clone the Chroma DB repository):
```
sudo yum install -y git
```

## 7. Clone the Chroma Core repository:
```
git clone https://github.com/chroma-core/chroma
```

## 8. Change your directory to the cloned Chroma directory:
```
cd chroma
```

## 9. Start the Docker service:
```
sudo systemctl start docker
```

## 10. Use Docker Compose to set up the Chroma DB:
```
sudo docker compose up -d --build
```

## 11. Verify that the Chroma DB is up and running:
Use a curl command to check the heartbeat:

```
curl http://localhost:8000/api/v1/heartbeat
```
You should see the output:
```
{"nanosecond heartbeat":1693440860301126345}
```

## 12. Check the running Docker containers:
You should see an entry for the chroma-server-1 container, serving on port 8000.
```
sudo docker ps

CONTAINER ID   IMAGE     COMMAND                  CREATED          STATUS          PORTS                                       NAMES
3af64879b16e   server    "uvicorn chromadb.ap…"   29 seconds ago   Up 28 seconds   0.0.0.0:8000->8000/tcp, :::8000->8000/tcp   chroma-server-1
```
