import * as React from "react";
import { Link } from "react-router-dom";

import { displayTimeInFromNowFormat } from "../utils/index";
import Table from "../ui/Table";

const parseApisAsArrayString = apis => {
  let list = [];
  if (apis) {
    for (let index = 0; index < apis.length; index++) {
      const api = apis[index];
      list.push(`${api.name}<${api.handler_type}>`);
    }
  }

  return list;
};

const parseArtifactsAsArrayString = artifacts => {
  let list = [];
  if (artifacts) {
    for (let index = 0; index < artifacts.length; index++) {
      const artifact = artifacts[index];
      list.push(`${artifact.name}<${artifact.artifact_type}>`);
    }
  }

  return list;
};

const BENTO_TABLE_HEADERS = [
  "BentoService(name:version)",
  "Age",
  "APIs",
  "Artifacts",
  ""
];
const BENTO_TABLE_RATIO = [6, 2, 4, 4, 1];

const BentoServiceTable = props => {
  const { bentos } = props;
  const parsedBentoServices = bentos.map(bento => {
    const metadata = bento.bento_service_metadata;
    const apis = parseApisAsArrayString(metadata.apis);
    const artifacts = parseArtifactsAsArrayString(metadata.artifacts);

    return [
      `${bento.name}:${bento.version}`,
      displayTimeInFromNowFormat(Number(metadata.created_at.seconds)),
      apis.join("\n"),
      artifacts.join("\n"),
      <Link to={`/repository/${bento.name}/${bento.version}`}>Detail</Link>
    ];
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
