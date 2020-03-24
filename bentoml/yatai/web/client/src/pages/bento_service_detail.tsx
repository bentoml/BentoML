import * as React from "react";

import { HttpRequestContainer } from "../utils/http_container";
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
      {({data, isLoading, error}) => {
        if (isLoading) {
          return <div>Loading...</div>
        }
        let displayBentoServiceDetail;
        if (error) {
          return <div>error</div>;
        }

        if (data && data && data.bento) {
          const bento = data.bento;

          displayBentoServiceDetail = (
            <div>
              <h4>created at date</h4>
              <h4>saved location</h4>
              <ApisTable apis={bento.bento_service_metadata.apis} />
              <ArtifactsTable
                artifacts={bento.bento_service_metadata.artifacts}
              />
              <EnvTable env={bento.bento_service_metadata.env} />
            </div>
          );
        } else {
          displayBentoServiceDetail = <div>grpc error</div>;
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
