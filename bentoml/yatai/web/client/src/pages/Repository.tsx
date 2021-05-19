import * as React from "react";
import { Link } from "react-router-dom";

import { getQueryObject } from "../utils";
import HttpRequestContainer from "../utils/HttpRequestContainer";
import BentoServiceTable from "../components/BentoServiceTable";
import { Section } from "../ui/Layout";
import LabelFilter from "../components/LabelFilter";
import QueryString from "qs";
import { query } from "express";

const DEFAULT_BENTO_SERVICE_LIMIT_PER_PAGE = 20;

const oldRepository = (props) => {
  const query = getQueryObject(props.location.search);
  const offset = Number(query.offset) || 0;
  return (
    <HttpRequestContainer
      url="/api/ListBento"
      method="get"
      params={{ limit: DEFAULT_BENTO_SERVICE_LIMIT_PER_PAGE, offset }}
    >
      {({ data }) => {
        let hasBento = false;
        let bentoDisplay;
        if (data && data.bentos) {
          hasBento = true;

          bentoDisplay = <BentoServiceTable bentos={data.bentos} />;
        } else {
          bentoDisplay = <Section>No more models found</Section>;
        }
        return (
          <Section>
            <LabelFilter />
            {bentoDisplay}
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                marginTop: "10px",
              }}
            >
              <div>
                {offset > 0 && (
                  <Link to={`/repository?offset=${offset - 10}`}>Previous</Link>
                )}
              </div>
              <div>
                {hasBento && (
                  <Link to={`/repository?offset=${offset + 10}`}>Next</Link>
                )}
              </div>
            </div>
          </Section>
        );
      }}
    </HttpRequestContainer>
  );
};

const RepositoryDisplay = (props) => {
  const { offset } = props;
  const params = {
    offset,
    limit: DEFAULT_BENTO_SERVICE_LIMIT_PER_PAGE,
  };
  return (
    <HttpRequestContainer url="/api/ListBento" method="get" params={params}>
      {({ data }) => {
        let hasBento = false;
        let bentoDisplay;
        if (data && data.bentos) {
          hasBento = true;

          bentoDisplay = <BentoServiceTable bentos={data.bentos} />;
        } else {
          bentoDisplay = <Section>No more models found</Section>;
        }
        return (
          <div>
            {bentoDisplay}
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                marginTop: "10px",
              }}
            >
              <div>
                {offset > 0 && (
                  <Link to={`/repository?offset=${offset - 10}`}>Previous</Link>
                )}
              </div>
              <div>
                {hasBento && (
                  <Link to={`/repository?offset=${offset + 10}`}>Next</Link>
                )}
              </div>
            </div>
          </div>
        );
      }}
    </HttpRequestContainer>
  );
};

const Repository = (props) => {
  const query = getQueryObject(props.location.search);
  const offset = Number(query.offset) || 0;
  let labelSelectors = [];
  return (
    <Section>
      <LabelFilter />
      <RepositoryDisplay offset={offset} labelSelectors={labelSelectors} />
    </Section>
  );
};

export default Repository;
