import * as React from "react";
import { Tag } from "@blueprintjs/core";

export interface ILabelProps {
  name: string;
  value: string;
}

const Label: React.FC<ILabelProps> = (props) => {
  const tagValue = `${props.name}:${props.value}`;
  return (
    <Tag key={tagValue} style={{ margin: 5 }}>
      {tagValue}
    </Tag>
  );
};

export default Label;
