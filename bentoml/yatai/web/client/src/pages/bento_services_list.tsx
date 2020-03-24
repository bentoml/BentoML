import * as React from 'react';
import { HttpRequestContainer } from '../utils/http_container';
import { BentoTable } from '../components/bento_service_table';

export const BentoServicesList = (props) => {
  const params = props.match.params;
  return (
    <div>
      <HttpRequestContainer
        url='/api/ListBento'
        method='get'
        params={{bento_name: params.name}}
      >
        {
          ({data, error}) => {
            if (data && data.bentos) {
              return (
                <div>
                  <h2>{params.name}</h2>
                  <BentoTable bentos={data.bentos} />
                </div>
              );
            } else {
              return (<div>ok</div>)
            }
          }
        }
      </HttpRequestContainer>
    </div>
  );
}