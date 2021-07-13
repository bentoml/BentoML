import * as React from "react";

import HttpRequestContainer from "../utils/HttpRequestContainer";
import DeploymentsTable from "../components/DeploymentsTable";
import BentoServiceTable from "../components/BentoServiceTable";
import { Section } from "../ui/Layout";

const HomePage = () => (
  <div>
    <HttpRequestContainer
      url="/api/ListDeployments"
      method="get"
      params={{ limit: 5 }}
    >
      {({ data }) => {
        let deploymentDisplay;
        if (data && data.deployments) {
          deploymentDisplay = (
            <DeploymentsTable deployments={data.deployments} />
          );
        } else {
          deploymentDisplay = (
            <a
              href="https://docs.bentoml.org/en/latest"
              target="_blank"
              rel="noopener noreferrer"
            >
              Learn about managing model serving deployments with BentoML
              <span role="img" aria-label="note">
                🔗
              </span>
            </a>
          );
        }

        return (
          <Section>
            <h2>Latest Deployments</h2>
            {deploymentDisplay}
          </Section>
        );
      }}
    </HttpRequestContainer>
    <HttpRequestContainer
      url="/api/ListBento"
      method="get"
      params={{ limit: 5 }}
    >
      {({ data }) => {
        if (data && data.bentos) {
          return (
            <Section>
              <h2>Latest Models</h2>
              <BentoServiceTable bentos={data.bentos} />
            </Section>
          );
        } else {
          return (
            <Section>
              <h2>No model found</h2>
              <a
                href="https://docs.bentoml.org/en/latest"
                target="_blank"
                rel="noopener noreferrer"
              >
                Learn about packaging ML model for serving with BentoML
                <span role="img" aria-label="note">
                  🔗
                </span>
              </a>
            </Section>
          );
        }
      }}
    </HttpRequestContainer>
  </div>
);

export default HomePage;
