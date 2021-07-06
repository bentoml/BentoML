import * as React from "react";
import * as moment from "moment";

import HttpRequestContainer from "../utils/HttpRequestContainer";
import EnvTable from "../components/BentoServiceDetail/EnvTable";
import ApisTable from "../components/BentoServiceDetail/ApisTable";
import ArtifactsTable from "../components/BentoServiceDetail/ArtifactsTable";
import { Section } from "../ui/Layout";
import LabelDetailSection from "../components/LabelDetailSection";
import DeleteConfirmation from "../components/BentoBundleDeleteConfirm";
import { findLastIndex } from "lodash";

const horizontalFlex = {
  display: "flex",
  alignItems: "flex-start",
  justifyContent: "space-between"
};

const TableHeader = styled(Row)({
  borderBottom: "2px solid #D8E1E8",
  fontWeight: 600,
  ":hover": {
    backgroundColor: "#F5F8FA",
  },
});

const BentoServiceDetail = (props) => {
  const params = props.match.params;

  return (
    <HttpRequestContainer
      url="/api/GetBento"
      params={{ bento_name: params.name, bento_version: params.version }}
    >
      {({ data }) => {
        let displayBentoServiceDetail;

        if (data && data.bento) {
          const bento = data.bento;

          displayBentoServiceDetail = (
            <div>
              <div style={horizontalFlex}>
                <div>
                  <p>
                    <b>Created at: </b>
                    {moment
                      .unix(Number(bento.bento_service_metadata.created_at.seconds))
                      .toDate()
                      .toLocaleString()}
                  </p>
                  <p>
                    <b>Storage: </b> {bento.uri.uri}
                  </p>
                  <LabelDetailSection
                    labels={bento.bento_service_metadata.labels}
                  />
                </div>
                <DeleteConfirmation></DeleteConfirmation>
              </div>
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
          <Section>
            <h2>
              {params.name}:{params.version}
            </h2>
            {displayBentoServiceDetail}
          </Section>
        );
      }}
    </HttpRequestContainer>
  );


};

export default BentoServiceDetail;
