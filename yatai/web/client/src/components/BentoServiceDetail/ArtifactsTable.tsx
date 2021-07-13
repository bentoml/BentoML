import * as React from "react";
import * as lodash from "lodash";
import Table from "../../ui/Table";
import { Section } from "../../ui/Layout";

interface IArtifactProps {
  name: string;
  artifact_type: string;
  metadata: any;
}

const ARTIFACTS_TABLE_HEADER = ["Name", "Type", "Metadata"];
const ARTIFACTS_TABLE_RATIO = [1, 2, 4];

const parseGrpcStructToObject = (structObject) => {
  /*
    grpc format for Struct format:
    key: {
      fields: {
        key: {
          structValue: {
            fields: {
              key: {
                listValue: {
                  values: [
                    {numberValue: 1},
                    {stringValue: 'abc'},
                  ]
                }
              },
              key: {
                stringValue: '123'
              },
            }
          }
        }
      }
    }
  */
  let result = {};
  if (structObject.fields) {
    structObject = structObject.fields;
  }
  const keyList = Object.keys(structObject);
  for (let index = 0; index < keyList.length; index++) {
    const key = keyList[index];
    const valueObject = structObject[key];
    const valueType = Object.keys(valueObject)[0];
    const value: any = Object.values(valueObject)[0];
    switch (valueType) {
      case "structValue":
        result[key] = parseGrpcStructToObject(value);
        break;
      case "listValue":
        result[key] = [];
        lodash.forEach(value, (item) => {
          result[key].push(parseGrpcStructToObject(item));
        });
        break;
      case "nullValue":
        result[key] = "null";
        break;
      default:
        result[key] = value;
    }
  }

  return result;
};

const artifactMetadataToTableContent = (artifactMetadata: any): string => {
  if (!artifactMetadata) {
    return "None";
  }
  let result = parseGrpcStructToObject(artifactMetadata);
  return JSON.stringify(result);
};

const ArtifactsTable: React.FC<{ artifacts: Array<IArtifactProps> }> = ({
  artifacts,
}) => {
  if (artifacts && artifacts.length > 0) {
    const artifactsTableContent = artifacts.map((artifact) => {
      return {
        content: [
          artifact.name,
          artifact.artifact_type,
          artifactMetadataToTableContent(artifact.metadata),
        ],
      };
    });
    return (
      <Section>
        <h2>Artifacts</h2>
        <Table
          content={artifactsTableContent}
          ratio={ARTIFACTS_TABLE_RATIO}
          header={ARTIFACTS_TABLE_HEADER}
        />
      </Section>
    );
  } else {
    return (
      <Section>
        <h2>Artifacts</h2>
        <p>No Artifacts</p>
      </Section>
    );
  }
};

export default ArtifactsTable;
