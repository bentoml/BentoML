import * as React from 'react';
import { getQueryObject } from '../utils';
import { HttpRequestContainer} from '../utils/http_container';
import { BentoTable } from '../components/bento_service_table';
import { Button } from '@blueprintjs/core';
import { Link } from 'react-router-dom';


export const Repository = (props) => {
  const query = getQueryObject(props.location.search);
  const offset = Number(query.offset) || 0;
  return (
    <HttpRequestContainer
      url='/api/ListBento'
      method='get'
      params={{limit: 10, offset}}
    >
      {
        ({data, isLoading, error}) => {
          if (isLoading) {
            return <div>Loading...</div>
          }
          if (error) {
            return <div>Error: {JSON.stringify(error)}</div>
          }
          let hasBento = false;
          let bentoDisplay;
          if (data && data.bentos) {
            hasBento = true;

            bentoDisplay = (
              <BentoTable bentos={data.bentos} />
            );
          } else {
            bentoDisplay = <div>No more models found</div>
          }
          return (
            <div>
              {bentoDisplay}
              <div>
                {
                  offset > 0 &&
                  <Link to={`/repository?offset=${offset-10}`}>Previous</Link>
                }
                {
                  hasBento && <Link to={`/repository?offset=${offset+10}`}>Next</Link>
                }
              </div>
            </div>
          )
        }
      }
    </HttpRequestContainer>
  )
};