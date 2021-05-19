import { TagInput } from "@blueprintjs/core";
import * as React from "react";

const LabelFilter: React.FC<any> = (props) => {
  return (
    <div>
      Label Selectors:
      <TagInput
        large={false}
        placeholder="Separate label selectors with commas..."
        values={["abc=123", "abc in (1,2,3)"]}
      />
    </div>
  );
};

export default LabelFilter;
