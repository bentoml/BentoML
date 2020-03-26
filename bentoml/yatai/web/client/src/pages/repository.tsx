import * as React from "react";

import { getQueryObject } from "../utils";
import HttpRequestContainer from "../utils/HttpRequestContainer";
import BentoServiceTable from "../components/BentoServiceTable";
import { Link } from "react-router-dom";

export const Repository = props => {
  const query = getQueryObject(props.location.search);
  const offset = Number(query.offset) || 0;
  return (
    <HttpRequestContainer
      url="/api/ListBento"
      method="get"
      params={{ limit: 10, offset }}
    >
      {({ data }) => {
        let hasBento = false;
        let bentoDisplay;
        if (data && data.bentos) {
          hasBento = true;

          bentoDisplay = <BentoServiceTable bentos={data.bentos} />;
        } else {
          bentoDisplay = <div>No more models found</div>;
        }
        return (
          <div>
            {bentoDisplay}
            <div>
              {offset > 0 && (
                <Link to={`/repository?offset=${offset - 10}`}>Previous</Link>
              )}
              {hasBento && (
                <Link to={`/repository?offset=${offset + 10}`}>Next</Link>
              )}
            </div>
          </div>
        );
      }}
    </HttpRequestContainer>
  );
};
