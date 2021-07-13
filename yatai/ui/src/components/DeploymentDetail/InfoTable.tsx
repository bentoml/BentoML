import * as React from "react";
import Table from "../../ui/Table";
import { displayTimeInFromNowFormat, displayTimeISOString } from "../../utils";
import { Link } from "react-router-dom";
import { Section } from "../../ui/Layout";

const displayTimestampForDeployment = (seconds: number) => {
  return `${displayTimeInFromNowFormat(seconds, true)} (${displayTimeISOString(
    seconds
  )})`;
};

const InfoTable = ({ deployment }) => {
  let endpointValues = "Not Available";
  if (deployment.state.state === "RUNNING" && deployment.state.info_json) {
    const infoJson = JSON.parse(deployment.state.info_json);
    endpointValues = infoJson.endpoints.join("\n");
  }
  const parsedInfo = [
    {
      content: [
        "Created at",
        displayTimestampForDeployment(Number(deployment.created_at.seconds)),
      ],
    },
    {
      content: [
        "Updated at",
        displayTimestampForDeployment(
          Number(deployment.last_updated_at.seconds)
        ),
      ],
    },
    {
      content: [
        "BentoService",
        <Link
          to={`/repository/${deployment.spec.bento_name}/${deployment.spec.bento_version}`}
        >
          {`${deployment.spec.bento_name}:${deployment.spec.bento_version}`}
        </Link>,
      ],
    },
    {
      content: ["Endpoint", endpointValues],
    },
  ];
  return (
    <Section>
      <h2>Info</h2>
      <Table content={parsedInfo} ratio={[1, 4]} />
    </Section>
  );
};

export default InfoTable;
