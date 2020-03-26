import * as React from "react";
import * as moment from "moment";

import HttpRequestContainer from "../utils/HttpRequestContainer";
import EnvTable from "../components/BentoServiceDetail/EnvTable";
import ApisTable from "../components/BentoServiceDetail/ApisTable";
import ArtifactsTable from "../components/BentoServiceDetail/ArtifactsTable";

export const BentoServiceDetail = props => {
  const params = props.match.params;

  return (
    <HttpRequestContainer
      url="/api/GetBento"
      params={{ bento_name: params.name, bento_version: params.version }}
    >
      {({ data }) => {
        let displayBentoServiceDetail;

        if (data && data && data.bento) {
          const bento = data.bento;

          displayBentoServiceDetail = (
            <div>
              <h4>
                Created at:{" "}
                {moment
                  .unix(Number(bento.bento_service_metadata.created_at.seconds))
                  .format("MM/DD/YYYY HH:mm:ss Z")}
              </h4>
              <h4>Storage: {bento.uri.uri}</h4>
              <ApisTable apis={bento.bento_service_metadata.apis} />
              <ArtifactsTable
                artifacts={bento.bento_service_metadata.artifacts}
              />
              <EnvTable env={bento.bento_service_metadata.env} />
            </div>
          );
        } else {
          displayBentoServiceDetail = <div>{JSON.stringify(data)}</div>;
        }

        return (
          <div>
            <h2>
              {params.name}:{params.version}
            </h2>
            {displayBentoServiceDetail}
          </div>
        );
      }}
    </HttpRequestContainer>
  );
};
