import * as React from "react";
import * as lodash from "lodash";

import Table from "../../ui/Table";

const parseKeyForDisplay = (key: string) => {
  return lodash.capitalize(key).replace("_", " ");
};

const ConfigurationTable = ({ spec }) => {
  const parsedConfiguration = [["Platform", spec.operator]];
  switch (spec.operator) {
    case "AWS_LAMBDA":
      lodash.each(spec.aws_lambda_operator_config, (value, key) => {
        if (key == "memory_size") {
          value = value + " MB";
        }
        switch (key) {
          case "memory_size":
            value = value + " MB";
          case "timeout":
            value = value + " seconds";
        }

        parsedConfiguration.push([parseKeyForDisplay(key), value]);
      });
      break;
    case "AWS_SAGEMAKER":
      lodash.each(spec.aws_lambda_operator_config, (value, key) => {
        parsedConfiguration.push([parseKeyForDisplay(key), value]);
      });
      break;
  }

  return (
    <div>
      <h2>Configuration</h2>
      <Table content={parsedConfiguration} ratio={[1, 4]} />
    </div>
  );
};

export default ConfigurationTable;
