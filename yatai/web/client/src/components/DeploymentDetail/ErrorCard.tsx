import * as React from "react";
import { Card } from "@blueprintjs/core";

const ErrorCard = ({ state }) => {
  if (["FAILED", "ERROR", "CRASH_LOOP_BACK_OFF"].indexOf(state.state) !== -1) {
    return (
      <Card elevation={3} style={{ backgroundColor: "orange" }}>
        {state.error_message}
      </Card>
    );
  }
  return null;
};

export default ErrorCard;
