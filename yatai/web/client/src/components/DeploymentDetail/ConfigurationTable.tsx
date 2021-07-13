import * as React from "react";
import * as lodash from "lodash";

import Table from "../../ui/Table";
import { Section } from "../../ui/Layout";

const parseKeyForDisplay = (key: string) => {
  return lodash.capitalize(key).replace("_", " ");
};

const ConfigurationTable = ({ spec }) => {
  const parsedConfiguration = [{ content: ["Platform", spec.operator] }];
  switch (spec.operator) {
    case "AWS_LAMBDA":
      lodash.each(spec.aws_lambda_operator_config, (value, key) => {
        switch (key) {
          case "memory_size":
            value = value + " MB";
            break;
          case "timeout":
            value = value + " seconds";
            break;
        }

        parsedConfiguration.push({ content: [parseKeyForDisplay(key), value] });
      });
      break;
    case "AWS_SAGEMAKER":
      lodash.each(spec.aws_lambda_operator_config, (value, key) => {
        parsedConfiguration.push({ content: [parseKeyForDisplay(key), value] });
      });
      break;
  }

  return (
    <Section>
      <h2>Configuration</h2>
      <Table content={parsedConfiguration} ratio={[1, 4]} />
    </Section>
  );
};

export default ConfigurationTable;
