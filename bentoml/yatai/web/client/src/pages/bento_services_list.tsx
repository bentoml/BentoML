import * as React from 'react';
import { HttpRequestContainer, DisplayHttpError } from '../utils/http_container';
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
          ({data, isLoading, error}) => {
            if (isLoading) {
              return <div>Loading...</div>
            }
            if (error) {
              return <DisplayHttpError error={error} />
            }
            if (data && data.bentos) {
              return (
                <div>
                  <h2>{params.name}</h2>
                  <BentoTable bentos={data.bentos} />
                </div>
              );
            } else {
              return (<div>{JSON.stringify(data)}</div>)
            }
          }
        }
      </HttpRequestContainer>
    </div>
  );
}
