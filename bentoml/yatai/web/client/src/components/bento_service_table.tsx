import * as React from 'react';
import * as moment from 'moment';
import { Link } from 'react-router-dom';
import { Button } from '@blueprintjs/core';
import { displayTimeInFromNowFormat } from '../utils/index';
import { TableContainer, TableHeader, Row, Cell } from '../ui/Table';

const parseApisAsArrayString = (apis) => {
  let list = [];
  if (apis) {
    for (let index = 0; index < apis.length; index++) {
      const api = apis[index];
      list.push(`${api.name}<${api.handler_type}>`)
    }
  }

  return list;
}

const parseArtifactsAsArrayString = (artifacts) => {
  let list = [];
  if (artifacts) {
    for (let index = 0; index < artifacts.length; index++) {
      const artifact = artifacts[index];
      list.push(`${artifact.name}<${artifact.artifact_type}>`);
    }
  }

  return list;
}


export const BentoTable = (props) => {
  const {bentos} = props;

  return (
    <TableContainer>
      <TableHeader>
        <Cell maxWidth={450} >BentoService(name:version)</Cell>
        <Cell maxWidth={150} >Age</Cell>
        <Cell maxWidth={400} >APIs</Cell>
        <Cell maxWidth={400} >Artifacts</Cell>
        <Cell maxWidth={200} ></Cell>
      </TableHeader>
      {
        bentos.map((bento, i) => {
          const metadata = bento.bento_service_metadata;
          const apis = parseApisAsArrayString(metadata.apis);
          const artifacts = parseArtifactsAsArrayString(metadata.artifacts);

          return (
            <Row key={i}>
              <Cell maxWidth={450} >{`${bento.name}:${bento.version}`}</Cell>
              <Cell maxWidth={150} >
                {
                  displayTimeInFromNowFormat(Number(metadata.created_at.seconds))
                }
              </Cell>
              <Cell maxWidth={400} >{apis.join('\n')}</Cell>
              <Cell maxWidth={400} >{artifacts.join('\n')}</Cell>
              <Cell maxWidth={200} >
                <div>
                  <Link to={`/repository/${bento.name}/${bento.version}`}>
                    Detail
                  </Link>
                </div>
              </Cell>
            </Row>
          )
        })
      }
    </TableContainer>
  )
};