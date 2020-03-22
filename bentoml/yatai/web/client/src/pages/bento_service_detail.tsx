import * as React from 'react';
import { Table, Column, Cell } from '@blueprintjs/table';

import { FetchContainer } from '../utils/index';
import { BentoServiceAPIs } from '../components/bento_service_api_table';
import { BentoServiceArtifacts } from '../components/bento_service_artifact_table';
import { BentoServiceEnvironments } from '../components/bento_service_env_table';

export const BentoServiceDetail = (props) => {
  const params = props.match.params;

  return (
    <FetchContainer
      url='/api/GetBento'
      params={{bento_name: params.name, bento_version: params.version}}
    >
      {
        (data, error) => {
          let displayBentoServiceDetail;
          if (error) {
            return (<div>error</div>);
          }

          if (data && data.data && data.data.bento) {
            console.log(data.data.bento);
            const bento = data.data.bento;

            displayBentoServiceDetail = (
              <div>
                <h2>{params.name}:{params.version}</h2>
                <h4>created at date</h4>
                <h4>saved location</h4>
                <BentoServiceAPIs apis={bento.bento_service_metadata.apis}/>
                <BentoServiceArtifacts
                  artifacts={bento.bento_service_metadata.artifacts}
                />
                <BentoServiceEnvironments env={bento.bento_service_metadata.env}/>
              </div>
            )
          } else {
            displayBentoServiceDetail = (
              <div>grpc error</div>
            )
          }

          return (
            <div>
              {displayBentoServiceDetail}
            </div>
          )
        }
      }
    </FetchContainer>
  )
};