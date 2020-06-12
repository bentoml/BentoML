import * as React from "react";
import { Link } from "react-router-dom";

import { displayTimeInFromNowFormat } from "../utils/index";
import Table from "../ui/Table";

const apisToDisplayFormat = (apis) => {
  if (!apis) {
    return "";
  }
  return apis.map((api) => `${api.name}<${api.input_type}>`).join("\n");
};

const artifactsToDisplayFormat = (artifacts) => {
  if (!artifacts) {
    return "";
  }
  return artifacts
    .map((artifact) => `${artifact.name}<${artifact.artifact_type}>`)
    .join("\n");
};

const BENTO_TABLE_HEADERS = [
  "BentoService(name:version)",
  "Age",
  "APIs",
  "Artifacts",
  "",
];
const BENTO_TABLE_RATIO = [6, 2, 4, 4, 1];

const BentoServiceTable = (props) => {
  const { bentos } = props;
  const parsedBentoServices = bentos.map((bento) => {
    const metadata = bento.bento_service_metadata;
    const apis = apisToDisplayFormat(metadata.apis);
    const artifacts = artifactsToDisplayFormat(metadata.artifacts);

    return {
      content: [
        `${bento.name}:${bento.version}`,
        displayTimeInFromNowFormat(Number(metadata.created_at.seconds)),
        apis,
        artifacts,
        <Link to={`/repository/${bento.name}/${bento.version}`}>Detail</Link>,
      ],
      link: `/repository/${bento.name}/${bento.version}`,
    };
  });

  return (
    <Table
      content={parsedBentoServices}
      ratio={BENTO_TABLE_RATIO}
      header={BENTO_TABLE_HEADERS}
    />
  );
};

export default BentoServiceTable;
