import * as React from "react";
import { Link } from "react-router-dom";
import * as lodash from "lodash";

import { displayTimeInFromNowFormat } from "../utils";
import Table from "../ui/Table";
import Label from "./Label";

const apisToDisplayFormat = (apis: any) => {
  if (!apis) {
    return "";
  }
  return apis.map((api: any) => `${api.name}<${api.input_type}>`).join("\n");
};

const artifactsToDisplayFormat = (artifacts) => {
  if (!artifacts) {
    return "";
  }
  return artifacts
    .map((artifact) => `${artifact.name}<${artifact.artifact_type}>`)
    .join("\n");
};

const labelsToDisplayFormat = (labels) => {
  if (!labels) {
    return "";
  }
  return (
    <div>
      {lodash.map(labels, (value, name) => {
        return <Label name={name} value={value} />;
      })}
    </div>
  );
};

const BENTO_TABLE_HEADERS = [
  "BentoService(name:version)",
  "Age",
  "Labels",
  "APIs",
  "Artifacts",
  "",
];
const BENTO_TABLE_RATIO = [5, 2, 5, 3, 3, 1];

const BentoServiceTable = (props) => {
  const { bentos } = props;
  const parsedBentoServices = bentos.map((bento) => {
    const metadata = bento.bento_service_metadata;
    if (lodash.isEmpty(metadata)) {
      // When a BentoBundle is created but bundle file upload is not finished, the
      // metadata field will be empty
      return {
        content: [
          `${bento.name}:${bento.version}`,
          null,
          null,
          null,
          null,
          <Link to={`/repository/${bento.name}/${bento.version}`}>Detail</Link>,
        ],
      };
    } else {
      const apis = apisToDisplayFormat(metadata.apis);
      const artifacts = artifactsToDisplayFormat(metadata.artifacts);
      const labels = labelsToDisplayFormat(metadata.labels);

      return {
        content: [
          `${bento.name}:${bento.version}`,
          displayTimeInFromNowFormat(Number(metadata.created_at.seconds)),
          labels,
          apis,
          artifacts,
          <Link to={`/repository/${bento.name}/${bento.version}`}>Detail</Link>,
        ],
        link: `/repository/${bento.name}/${bento.version}`,
      };
    }
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
