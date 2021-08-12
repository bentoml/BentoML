import * as React from "react";
import { Link } from "react-router-dom";
import * as lodash from "lodash";

import { displayTimeInFromNowFormat } from "../utils";
import Table from "../containers/Table";
import Label from "./Label";
import Searchbar from "./Searchbar";
import { YataiToaster } from "../utils/Toaster";
import { Intent } from "@blueprintjs/core";

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

const labelKeys = (labels) => {
  if (!labels) {
    return "";
  }
  return lodash.map(labels, (value, name) => name.toLowerCase());
};

const BENTO_TABLE_HEADERS = [
  "BentoService(name:version)",
  "Age",
  "Labels",
  "APIs",
  "Artifacts",
  ""
];
const BENTO_TABLE_RATIO = [5, 2, 5, 3, 3, 1];

const BentoServiceTable = (props) => {
  const { bentos } = props;
  const [filteredBentos, setFilteredBentos] = React.useState(bentos);

  const filterBentoServices = (filters: string) => {
    // prettier-ignore
    let jsonFilter = '{';
    filters = filters.trim();
    const filterArray = filters.split(/\s+/);
    filterArray.forEach((filterElement) => {
      const filter: string[] = filterElement.split(":");
      if (filter[0] !== "") {
        jsonFilter += `"${filter[0]}":"${filter[1]}",`;
      }
    });
    if (jsonFilter.length > 1) {
      jsonFilter = jsonFilter.slice(0, -1);
    }
    // prettier-ignore
    jsonFilter += '}';
    const filter = JSON.parse(jsonFilter);

    setFilteredBentos(
      bentos.filter((bento) => {
        const metadata = bento.bento_service_metadata;
        if (!lodash.isEmpty(metadata)) {
          const name = bento.name;
          const apis = apisToDisplayFormat(metadata.apis);
          const artifacts = artifactsToDisplayFormat(metadata.artifacts);
          const labels = labelKeys(metadata.labels);
          const bentoJson = {
            name: name,
            api: apis,
            artifacts: artifacts,
            labelKey: labels
          };

          for (let key in filter) {
            if (bentoJson[key] === undefined) {
              const toastState = {
                message: "Key not found, be sure to use proper syntax.",
                intent: Intent.DANGER
              };
              YataiToaster.show({ ...toastState });
            } else if (key === "labelKey") {
              return bentoJson[key].includes(filter[key].toLowerCase());
            } else if (
              !bentoJson[key].toLowerCase().includes(filter[key].toLowerCase())
            ) {
              return false;
            }
          }
          return true;
        }
        return false;
      })
    );
  };

  const parsedBentoServices = filteredBentos.map((bento) => {
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
          <Link to={`/repository/${bento.name}/${bento.version}`}>Detail</Link>
        ]
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
          <Link to={`/repository/${bento.name}/${bento.version}`}>Detail</Link>
        ],
        link: `/repository/${bento.name}/${bento.version}`
      };
    }
  });

  return (
    <div>
      <Searchbar handleFilter={filterBentoServices.bind(this)} />
      <Table
        content={parsedBentoServices}
        ratio={BENTO_TABLE_RATIO}
        header={BENTO_TABLE_HEADERS}
      />
    </div>
  );
};

export default BentoServiceTable;
