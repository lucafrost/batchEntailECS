#!/bin/bash

# build, tag, and push to AWS ECR

if [ $# -ge 1 ]
then
    echo "tag specified $1"
    tag_read=$1
else
    echo "no tag specified. defaulting to latest"
    tag_read="fix-1"
fi

# prerequisites: docker (obviously), awscli configured, and a repo on ECR set up called 'awsgpu'

docker_exec="sudo docker"
# docker_exec="sudo docker"  # depending on your setup you may want this

tag="batchentailment:${tag_read}"

$docker_exec build --tag $tag . --no-cache 

echo "build over at `date`"

while true; do
    read -p "continue with push?" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

echo "continuing to push..."

account="356359304549"  # fill in
region="eu-west-2"  # for example

amazon_url="${account}.dkr.ecr.${region}.amazonaws.com"

echo "logging into AWS for docker..."

aws ecr get-login-password --region ${region} | $docker_exec login --username AWS --password-stdin $amazon_url

$docker_exec tag $tag "$amazon_url/$tag"
echo "pushing..."
$docker_exec push "$amazon_url/$tag"
echo "done at `date`"