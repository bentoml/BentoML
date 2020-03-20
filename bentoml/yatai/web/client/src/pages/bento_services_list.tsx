import * as React from 'react';
import { FetchContainer } from '../utils/index';
import { BentoTable } from '../components/bento_table';

export const BentoServicesList = (props) => {
  const params = props.match.params;
  return (
    <div>
      <FetchContainer
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
      </FetchContainer>
    </div>
  );
}