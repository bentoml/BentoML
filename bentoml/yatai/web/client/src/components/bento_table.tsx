import * as React from 'react';
import * as moment from 'moment';
import { Link } from 'react-router-dom';
import { Button } from '@blueprintjs/core';
import { Table, Cell, Column } from '@blueprintjs/table';
import { displayTimeInFromNowFormat } from '../utils/index';


export const BentoTable = (props) => {
  const {bentos} = props;

  const renderBentoTag = (rowIndex: number) => {
    const bento = bentos[rowIndex];
    return (
      <Cell>{`${bento.name}:${bento.version}`}</Cell>
    )
  };

  const renderAge = (rowIndex: number) => {
    const bento = bentos[rowIndex];
    const metadata = bento.bento_service_metadata;

    return (
      <Cell>{displayTimeInFromNowFormat(Number(metadata.created_at.seconds))}</Cell>
    )
  };

  const renderAPIs = (rowIndex: number) => {
    const bento = bentos[rowIndex];
    const metadata = bento.bento_service_metadata;

    let apis = [];
    for (let index = 0; index < metadata.apis.length; index++) {
      const api = metadata.apis[index];
      apis.push(`${api.name}<${api.handler_type}>`)
    }

    return (
      <Cell>{apis.join()}</Cell>
    )
  };

  const renderArtifacts = (rowIndex: number) => {
    const bento = bentos[rowIndex];
    const metadata = bento.bento_service_metadata;

    let artifacts = [];
    for (let index = 0; index < metadata.artifacts.length; index++) {
      const artifact = metadata.artifacts[index];
      artifacts.push(`${artifact.name}<${artifact.artifact_type}>`);
    }

    return (
      <Cell key='artifacts-cell'>{'okkj,w'}</Cell>
    )
  };

  const renderActions = (rowIndex: number) => {
    const bento = bentos[rowIndex];
    const detailLink = `repository/${bento.name}/${bento.version}`;

    return (
      <Cell>
        <div>
          <Link to={detailLink}><Button>Detail</Button></Link>
          <Button>Download</Button>
        </div>
      </Cell>
    )
  };

  return (
    <Table numRows={bentos.length}>
      <Column name="BentoService(name:version)" cellRenderer={renderBentoTag}/>
      <Column name="Age" cellRenderer={renderAge}/>
      <Column name="APIs" cellRenderer={renderAPIs}/>
      {/* <Column name="Artifacts" cellRenderer={renderArtifacts} /> */}
      <Column name='' cellRenderer={renderActions} />
    </Table>
  );
};