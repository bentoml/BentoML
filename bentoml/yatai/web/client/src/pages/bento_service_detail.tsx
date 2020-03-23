import * as React from "react";

import { FetchContainer } from "../utils/index";
import EnvTable from "../components/BentoServiceDetail/EnvTable";
import ApisTable from "../components/BentoServiceDetail/ApisTable";
import ArtifactsTable from "../components/BentoServiceDetail/ArtifactsTable";

export const BentoServiceDetail = props => {
  const params = props.match.params;

  return (
    <FetchContainer
      url="/api/GetBento"
      params={{ bento_name: params.name, bento_version: params.version }}
    >
      {(data, error) => {
        let displayBentoServiceDetail;
        if (error) {
          return <div>error</div>;
        }

        if (data && data.data && data.data.bento) {
          console.log(data.data.bento);
          const bento = data.data.bento;

          displayBentoServiceDetail = (
            <div>
              <h2>
                {params.name}:{params.version}
              </h2>
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

        return <div>{displayBentoServiceDetail}</div>;
      }}
    </FetchContainer>
  );
};
