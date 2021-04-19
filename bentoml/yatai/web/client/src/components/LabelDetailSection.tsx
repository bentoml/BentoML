import * as React from "react";
import * as lodash from "lodash";
import Label from "./Label";

export interface ILabelDetailSectionProps {
  labels: any;
}

const LabelDetailSection: React.FC<ILabelDetailSectionProps> = (props) => {
  return (
    <p>
      <b>Labels: </b>{" "}
      {lodash.map(props.labels, (value, name) => {
        return <Label name={name} value={value} />;
      })}
    </p>
  );
};

export default LabelDetailSection;
