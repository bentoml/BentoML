#!/bin/bash

ORG_NAME=$1
REPO_NAME=$2
WORKFLOW_NAME=$3

if [[ -z $WORKFLOW_NAME || -z $REPO_NAME || -z $ORG_NAME ]]; then
  echo "Usage: ./workflow_cleanup.sh <USER/ORG NAME> <REPO NAME> <WORKFLOW NAME>"
  exit 1
fi

if ! gh auth status 2>&1 >/dev/null | grep "Logged in to github.com"; then
  echo "Script requires logged-in gh CLI user."
  echo "Install gh CLI according to your OS."
  exit 1
fi

JQ_SEARCH=".workflows[] | select(.name == \"$WORKFLOW_NAME\") | \"\\(.id)\""

echo "Searching for workflow..."
WORKFLOW_ID=$(gh api "repos/$ORG_NAME/$REPO_NAME/actions/workflows?per_page=100" | jq -r "$JQ_SEARCH")

if [ -z "$WORKFLOW_ID" ]; then
  printf "Workflow not found!\nCheck your spelling!"
  exit 1;
else
  echo "...done!"
fi

echo "Retrieving runs..."
WORKFLOW_RUNS=$(gh api "repos/$ORG_NAME/$REPO_NAME/actions/workflows/$WORKFLOW_ID/runs?per_page=100")
echo "done!"

RUN_COUNT=$(echo "$WORKFLOW_RUNS" | jq -r ".total_count")

read -r -p "Are you sure you want to delete all $RUN_COUNT runs? [y/N] " response


if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
  DELETED=0
  LEFT=1
  while [[ $LEFT -gt "0" ]]; do
    # Be lazy and try to avoid fetching stale results
    sleep 1

    BATCH=$(gh api "repos/$ORG_NAME/$REPO_NAME/actions/workflows/$WORKFLOW_ID/runs?per_page=100")
    LEFT=$(echo "$BATCH" | jq -r ".total_count")
    echo "Deleting up to 100 of $LEFT"

    echo "$BATCH" \
      | jq -r '.workflow_runs[] | "\(.id)"' \
      | xargs -n1 -I % gh api repos/"$ORG_NAME"/"$REPO_NAME"/actions/runs/% -X DELETE

    ((DELETED=DELETED+LEFT))
    echo "done!"
  done;

  echo ""
  echo "All Done! Deleted $DELETED runs from \"$WORKFLOW_NAME\""
else
  echo "Writing preliminary data to /tmp/workflow_info.json and aborting!"
  echo "$WORKFLOW_RUNS" > /tmp/workflow_info.json
fi
